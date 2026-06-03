"""Flask Web Application for Audio Time Series Analysis."""

import os
import uuid
import numpy as np
from flask import Flask, render_template, request, redirect, url_for

# Import our audio analysis package
from audiots import loader, features, dynamics, discovery, analysis, prediction, band_analysis, visualization
from audiots import similarity, similarity_viz, discovery_viz

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)


def analyze_audio(filepath1, filepath2=None, forecast_horizon=20, n_mels=128):
    """Perform complete audio analysis."""
    results = {}
    
    # Load audio
    y1, sr = loader.load_audio(filepath1, target_sr=16000)
    results['audio_info'] = {
        'duration': len(y1) / sr,
        'sample_rate': sr,
        'samples': len(y1)
    }

    # Store raw audio for downstream use (visualization, etc.)
    results['_y1'] = y1
    results['_sr'] = sr
    
    # Feature extraction
    t, y = features.compute_waveform(y1, sr)
    results['waveform'] = {'t': t, 'y': y}
    
    freqs, mag = features.compute_fft(y1, sr)
    results['fft'] = {'freqs': freqs, 'mag': mag}
    
    f_stft, t_stft, spec_stft = features.compute_stft(y1, sr)
    results['stft'] = {'freqs': f_stft, 'times': t_stft, 'spec': spec_stft}
    
    mel_freqs, mel_times, mel_spec = features.compute_mel_spectrogram(y1, sr, n_mels=n_mels)
    results['mel'] = {'freqs': mel_freqs, 'times': mel_times, 'spec': mel_spec}

    # ---- Dynamics Layer: Trend Analysis ----
    # Inserted between Feature Extraction and Time Series Analysis
    dyn1 = dynamics.extract_dynamics(y1, sr, window_size=0.5, hop_size=0.25)
    results['dynamics'] = dyn1
    results['dynamics_segments'] = dynamics.detect_structural_segments(dyn1)

    # Time series analysis
    lags, acf_vals, ci = analysis.compute_acf(y[:min(len(y), 4000)], nlags=40)
    _, pacf_vals, _ = analysis.compute_pacf(y[:min(len(y), 4000)], nlags=40)
    results['acf_pacf'] = {'lags': lags, 'acf': acf_vals, 'pacf': pacf_vals, 'ci': ci}
    
    periodicity = analysis.analyze_periodicity(y1, sr)
    results['periodicity'] = periodicity
    
    complexity = analysis.analyze_complexity(y1)
    results['complexity'] = complexity
    
    flatness = analysis.compute_spectral_flatness(mag)
    results['spectral_flatness'] = flatness
    
    # Prediction
    results['predictions'] = prediction.run_all_predictions(mel_spec, forecast_horizon=forecast_horizon, verbose=False)
    
    # Band analysis
    band_results = band_analysis.analyze_band_predictability(mel_spec, forecast_horizon=forecast_horizon)
    results['band_results'] = band_results
    results['band_summary'] = band_analysis.compute_band_error_summary(band_results)
    results['predictability_rank'] = band_analysis.get_predictability_rank(results['band_summary'])
    
    # Dual audio analysis — full similarity pipeline
    if filepath2:
        y2, sr2 = loader.load_audio(filepath2, target_sr=sr)

        # ---- Discovery: multi-dimensional exploration (replaces scoring) ----
        results['_y2'] = y2
        disc_report = discovery.explore(
            y1, sr, y2, sr2,
            window_size=0.5, hop_size=0.25, verbose=True,
        )
        results['discovery'] = disc_report

        # Legacy DTW kept for backward compatibility
        try:
            from dtw import dtw
            from scipy.spatial.distance import euclidean
            mel2_freqs, mel2_times, mel2_spec = features.compute_mel_spectrogram(y2, sr2, n_mels=n_mels)
            mel1_mean = np.mean(mel_spec, axis=0)[:100]
            mel2_mean = np.mean(mel2_spec, axis=0)[:100]
            dtw_result = dtw(mel1_mean.reshape(-1, 1), mel2_mean.reshape(-1, 1), dist_method=euclidean)
            results['dtw'] = {
                'distance': dtw_result.distance,
                'similarity': 1 - dtw_result.distance / (len(mel1_mean) * np.std(mel1_mean)),
                'path': dtw_result.index.tolist(),
                'x': mel1_mean, 'y': mel2_mean,
            }
        except ImportError:
            results['dtw'] = {'error': 'dtw library not installed'}

    return results


@app.route('/', methods=['GET', 'POST'])
def index():
    """Main page with upload form."""
    if request.method == 'POST':
        # Generate unique ID for this analysis
        task_id = str(uuid.uuid4())[:8]
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], task_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save uploaded files
        filepath1 = None
        filepath2 = None
        
        if 'audio1' in request.files and request.files['audio1'].filename != '':
            file = request.files['audio1']
            filename = secure_filename(file.filename)
            filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_1_{filename}")
            file.save(filepath1)
        
        if 'audio2' in request.files and request.files['audio2'].filename != '':
            file = request.files['audio2']
            filename = secure_filename(file.filename)
            filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_2_{filename}")
            file.save(filepath2)
        
        # Use synthetic audio if no file uploaded
        if not filepath1:
            y1, sr = loader.generate_sample_audio(duration=5.0)
            filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_synthetic.wav")
            import soundfile as sf
            sf.write(filepath1, y1, sr)
        
        # Get parameters
        forecast_horizon = int(request.form.get('forecast_horizon', 20))
        n_mels = int(request.form.get('n_mels', 128))
        
        # Perform analysis
        results = analyze_audio(filepath1, filepath2, forecast_horizon, n_mels)
        
        # Generate plots
        plot_files = visualization.generate_report_plots(results, output_dir)

        # Generate discovery plots if dual-audio analysis was performed
        if results.get('discovery'):
            disc = results['discovery']
            dyn_a = results.get('dynamics')
            # Re-extract dyn2 for visualization
            dyn_b = None
            if results.get('_y2') is not None and results.get('_sr') is not None:
                dyn_b = dynamics.extract_dynamics(results['_y2'], results['_sr'],
                                                  window_size=0.5, hop_size=0.25)

            disc_plots = discovery_viz.generate_discovery_report_plots(
                disc,
                output_dir=os.path.join(output_dir, 'discovery'),
                y1=results.get('_y1'), y2=results.get('_y2'),
                sr=results.get('_sr'),
                dyn1=dyn_a, dyn2=dyn_b,
            )
            plot_files += [f"discovery/{p}" for p in disc_plots]

        # Generate dynamics/trend plots
        if results.get('dynamics'):
            dyn_dir = os.path.join(output_dir, 'dynamics')
            os.makedirs(dyn_dir, exist_ok=True)
            dyn = results['dynamics']
            seg = results.get('dynamics_segments')
            visualization.plot_dynamics_trends(
                dyn, segments=seg,
                save_path=os.path.join(dyn_dir, 'trends.png'),
            )
            visualization.plot_dynamics_summary(
                dyn, segments=seg,
                save_path=os.path.join(dyn_dir, 'summary.png'),
            )
            plot_files.append('dynamics/trends.png')
            plot_files.append('dynamics/summary.png')

        # Create serializable version of discovery report (strip raw arrays)
        if results.get('discovery'):
            disc = results['discovery']

            def _profile_to_dict(p):
                if p is None:
                    return None
                return {
                    'rhythm_signature': p.rhythm_signature,
                    'energy_profile': p.energy_profile,
                    'timbre_quality': p.timbre_quality,
                    'standout_features': p.standout_features,
                }

            def _discovery_to_dict(d):
                return {
                    'title': d.title,
                    'dimension': d.dimension,
                    'discovery_type': d.discovery_type,
                    'summary': d.summary,
                    'n_matches': len(d.segment_matches),
                    'avg_confidence': d.meta.get('avg_confidence', 0),
                    'matches': [
                        {
                            'a_start': sm.a.start, 'a_end': sm.a.end,
                            'b_start': sm.b.start, 'b_end': sm.b.end,
                            'confidence': sm.evidence.confidence,
                            'method': sm.evidence.method,
                        }
                        for sm in d.segment_matches[:10]
                    ],
                }

            results['discovery_serializable'] = {
                'audio_a_profile': _profile_to_dict(disc.audio_a_profile),
                'audio_b_profile': _profile_to_dict(disc.audio_b_profile),
                'audio_a_motifs_count': len(disc.audio_a_motifs),
                'audio_b_motifs_count': len(disc.audio_b_motifs),
                'n_discoveries': len(disc.discoveries),
                'n_contrasts': len(disc.contrasts),
                'overview': disc.overview,
                'discoveries': [_discovery_to_dict(d) for d in disc.discoveries],
                'contrasts': [_discovery_to_dict(c) for c in disc.contrasts],
                'params': disc.params,
            }

        # Create serializable version of dynamics (strip raw arrays)
        if results.get('dynamics'):
            from audiots.dynamics import summarize_dynamics
            dyn_summary = summarize_dynamics(results['dynamics'])
            seg = results.get('dynamics_segments', {})
            results['dynamics_serializable'] = {
                'summary': dyn_summary,
                'n_climax': len(seg.get('climax_indices', [])),
                'n_calm': len(seg.get('calm_indices', [])),
                'n_buildup': len(seg.get('buildup_indices', [])),
                'n_transition': len(seg.get('transition_indices', [])),
            }
            if results.get('dynamics_similarity'):
                ds = results['dynamics_similarity']
                results['dynamics_serializable']['similarity'] = {
                    'global': ds['global_dynamics_similarity'],
                    'dominant_trend': ds['dominant_trend'],
                    'coherence': ds['structural_coherence'],
                    'per_trend': {
                        k: {'structural_score': v['structural_score']}
                        for k, v in ds['per_trend'].items()
                    },
                }

        # Store results
        results['task_id'] = task_id
        results['plot_files'] = plot_files
        results['params'] = {
            'forecast_horizon': forecast_horizon,
            'n_mels': n_mels,
            'audio1_name': os.path.basename(filepath1),
            'audio2_name': os.path.basename(filepath2) if filepath2 else None
        }
        
        # Save results to JSON (strip internal/large data first)
        serializable = {k: v for k, v in results.items()
                        if not k.startswith('_') and k not in (
                            'similarity', 'discovery',
                            'dynamics', 'dynamics_2',
                            'dynamics_segments', 'dynamics_segments_2',
                            'dynamics_similarity',
                        )}
        if results.get('similarity_serializable'):
            serializable['similarity'] = results['similarity_serializable']
        if results.get('discovery_serializable'):
            serializable['discovery'] = results['discovery_serializable']
        if results.get('dynamics_serializable'):
            serializable['dynamics'] = results['dynamics_serializable']

        import json
        with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as f:
            json.dump(serializable, f, default=str, ensure_ascii=False)
        
        return redirect(url_for('results', task_id=task_id))
    
    return render_template('index.html')


def secure_filename(filename):
    """Simple secure filename function."""
    import re
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', filename)


@app.route('/results/<task_id>')
def results(task_id):
    """Display analysis results."""
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], task_id)
    
    if not os.path.exists(output_dir):
        return "分析任务不存在", 404
    
    # Load results
    import json
    with open(os.path.join(output_dir, 'results.json'), 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Get plot URLs
    plot_files = results.get('plot_files', [])
    plot_urls = {os.path.splitext(f)[0]: url_for('static', filename=f'outputs/{task_id}/{f}') 
                 for f in plot_files}
    
    return render_template('results.html', 
                          results=results, 
                          plot_urls=plot_urls,
                          task_id=task_id)


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for analysis."""
    if 'audio1' not in request.files:
        return {'error': 'No audio file provided'}, 400
    
    file = request.files['audio1']
    if file.filename == '':
        return {'error': 'No selected file'}, 400
    
    task_id = str(uuid.uuid4())[:8]
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], task_id)
    os.makedirs(output_dir, exist_ok=True)
    
    filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{secure_filename(file.filename)}")
    file.save(filepath1)
    
    forecast_horizon = int(request.form.get('forecast_horizon', 20))
    n_mels = int(request.form.get('n_mels', 128))
    
    try:
        results = analyze_audio(filepath1, None, forecast_horizon, n_mels)
        visualization.generate_report_plots(results, output_dir)
        return {'task_id': task_id, 'message': 'Analysis completed'}
    except Exception as e:
        return {'error': str(e)}, 500


@app.route('/api/results/<task_id>')
def api_results(task_id):
    """API endpoint to get results."""
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], task_id)
    
    if not os.path.exists(output_dir):
        return {'error': 'Task not found'}, 404
    
    import json
    with open(os.path.join(output_dir, 'results.json'), 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    return results


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)