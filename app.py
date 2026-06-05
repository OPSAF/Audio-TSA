"""Flask Web Application for Audio Time Series Analysis."""

import os
import sys
import uuid
import json
import time
import threading
import numpy as np
from queue import Queue
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from flask import Flask, render_template, request, redirect, url_for, Response, stream_with_context, jsonify

# Import our audio analysis package
from audiots import loader, features, dynamics, volatility, model_analysis, discovery, unsupervised, analysis, prediction, band_analysis, visualization
from audiots import similarity, similarity_viz, discovery_viz

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Store for analysis progress
analysis_progress = {}

# Available analysis options
ANALYSIS_OPTIONS = [
    {'id': 'features', 'name': '特征提取', 'description': '波形、频谱、STFT、Mel、MFCC', 'default': True},
    {'id': 'dynamics', 'name': '动态趋势分析', 'description': '能量、亮度、复杂度、节奏', 'default': True},
    {'id': 'dynamics_analysis', 'name': '音频动态分析', 'description': 'Trend Layer + Volatility Layer (ARCH/GARCH)', 'default': True},
    {'id': 'model_analysis', 'name': '模型结构分析', 'description': 'ARIMA/HMM/LSTM/Transformer 结构侦探', 'default': True},
    {'id': 'timeseries', 'name': '时序分析', 'description': 'ACF、PACF、周期性、复杂度', 'default': True},
    {'id': 'unsupervised', 'name': '无监督模式发现', 'description': '聚类、 motif检测', 'default': True},
    {'id': 'prediction', 'name': '机器学习预测', 'description': 'ARIMA、HMM、LSTM、Transformer', 'default': True},
    {'id': 'band', 'name': '频带分析', 'description': '频率带可预测性评估', 'default': True},
    {'id': 'comparison', 'name': '双音频对比', 'description': '相似度分析、多维探索', 'default': False},
    {'id': 'visualization', 'name': '可视化', 'description': '生成分析图表', 'default': True},
]


def parse_analysis_options(form_data, has_dual_audio):
    """Parse analysis options from form data."""
    selected = []
    
    for opt in ANALYSIS_OPTIONS:
        key = f'analysis_{opt["id"]}'
        if key in form_data and form_data[key] == 'on':
            selected.append(opt['id'])
    
    # Handle comparison option - only available with dual audio
    if 'comparison' in selected and not has_dual_audio:
        selected.remove('comparison')
    
    # If no options selected, use defaults
    if not selected:
        for opt in ANALYSIS_OPTIONS:
            if opt['default']:
                if opt['id'] == 'comparison' and not has_dual_audio:
                    continue
                selected.append(opt['id'])
    
    return selected


class ProgressWriter:
    """Capture stdout/stderr and send printed text into the task log."""
    def __init__(self, task_id, level='info'):
        self.task_id = task_id
        self.level = level
        self._buffer = ''

    def write(self, text):
        if not text:
            return
        self._buffer += text
        while '\n' in self._buffer:
            line, self._buffer = self._buffer.split('\n', 1)
            if line.strip():
                log_progress(self.task_id, line.strip(), level=self.level)

    def flush(self):
        if self._buffer.strip():
            log_progress(self.task_id, self._buffer.strip(), level=self.level)
            self._buffer = ''


def log_progress(task_id, message, level='info'):
    """Log progress message for a task."""
    if task_id not in analysis_progress:
        analysis_progress[task_id] = {'logs': [], 'status': 'running', 'progress': 0}
    
    timestamp = time.strftime('%H:%M:%S')
    analysis_progress[task_id]['logs'].append({
        'time': timestamp,
        'message': message,
        'level': level
    })
    
    # Keep only last 100 logs
    if len(analysis_progress[task_id]['logs']) > 100:
        analysis_progress[task_id]['logs'] = analysis_progress[task_id]['logs'][-100:]


def update_progress(task_id, progress, status=None):
    """Update progress percentage for a task."""
    if task_id not in analysis_progress:
        analysis_progress[task_id] = {'logs': [], 'status': 'running', 'progress': 0}
    
    analysis_progress[task_id]['progress'] = progress
    if status:
        analysis_progress[task_id]['status'] = status


def analyze_audio_with_progress(task_id, filepath1, filepath2=None, forecast_horizon=20, n_mels=128, analysis_options=None):
    """Perform complete audio analysis with progress logging."""
    if analysis_options is None:
        analysis_options = [opt['id'] for opt in ANALYSIS_OPTIONS if opt['default']]
    
    results = {}
    dual_audio = filepath2 is not None
    dyn1 = None
    vol1 = None

    # Calculate progress steps
    total_steps = len(analysis_options) + 2  # +2 for loading and finalizing
    current_step = 0
    
    # Load audio
    log_progress(task_id, f"加载音频文件: {os.path.basename(filepath1)}")
    y1, sr = loader.load_audio(filepath1, target_sr=16000)
    results['audio_info'] = {
        'duration': len(y1) / sr,
        'sample_rate': sr,
        'samples': len(y1)
    }
    log_progress(task_id, f"音频信息: 时长={len(y1)/sr:.2f}s, 采样率={sr}Hz, 样本数={len(y1)}")
    
    current_step += 1
    update_progress(task_id, int(current_step / total_steps * 100))

    # Store raw audio for downstream use
    results['_y1'] = y1
    results['_sr'] = sr
    
    # ============================================================
    # Feature Extraction
    # ============================================================
    if 'features' in analysis_options:
        log_progress(task_id, "=" * 50, 'divider')
        log_progress(task_id, "PHASE 1: 特征提取", 'phase')
        log_progress(task_id, "=" * 50, 'divider')
        
        log_progress(task_id, "[1.1] 提取波形...")
        t, y = features.compute_waveform(y1, sr)
        results['waveform'] = {'t': t, 'y': y}
        
        log_progress(task_id, "[1.2] 计算FFT...")
        freqs, mag = features.compute_fft(y1, sr)
        results['fft'] = {'freqs': freqs, 'mag': mag}
        
        log_progress(task_id, "[1.3] 计算STFT...")
        f_stft, t_stft, spec_stft = features.compute_stft(y1, sr)
        results['stft'] = {'freqs': f_stft, 'times': t_stft, 'spec': spec_stft}
        
        log_progress(task_id, "[1.4] 计算Mel频谱图...")
        mel_freqs, mel_times, mel_spec = features.compute_mel_spectrogram(y1, sr, n_mels=n_mels)
        results['mel'] = {'freqs': mel_freqs, 'times': mel_times, 'spec': mel_spec}
        
        log_progress(task_id, f"[完成] Mel形状: {mel_spec.shape}")
        
        current_step += 1
        update_progress(task_id, int(current_step / total_steps * 100))
    
    # ============================================================
    # Dynamics Layer: Trend Analysis
    # ============================================================
    dyn1 = None
    if 'dynamics' in analysis_options:
        log_progress(task_id, "=" * 50, 'divider')
        log_progress(task_id, "PHASE 1.5: 动态趋势分析", 'phase')
        log_progress(task_id, "=" * 50, 'divider')
        
        log_progress(task_id, "[Dyn] 提取动态特征 (能量、亮度、复杂度、节奏)...")
        dyn1 = dynamics.extract_dynamics(y1, sr, window_size=0.5, hop_size=0.25)
        results['dynamics'] = dyn1
        results['dynamics_segments'] = dynamics.detect_structural_segments(dyn1)
        log_progress(task_id, "[完成] 动态趋势分析完成")
        
        if dual_audio and filepath2:
            y2, _ = loader.load_audio(filepath2, target_sr=sr)
            results['_y2'] = y2

        current_step += 1
        update_progress(task_id, int(current_step / total_steps * 100))

    # ============================================================
    # Dynamics Analysis: Trend Layer + Volatility Layer
    # ============================================================
    if 'dynamics_analysis' in analysis_options:
        # Ensure dynamics are extracted first (if not already done)
        if dyn1 is None:
            log_progress(task_id, "=" * 50, 'divider')
            log_progress(task_id, "PHASE 1.5a: 动态趋势分析 (Trend Layer)", 'phase')
            log_progress(task_id, "=" * 50, 'divider')

            log_progress(task_id, "[Dyn] 提取动态特征 (能量、亮度、复杂度、节奏)...")
            dyn1 = dynamics.extract_dynamics(y1, sr, window_size=0.5, hop_size=0.25)
            results['dynamics'] = dyn1
            results['dynamics_segments'] = dynamics.detect_structural_segments(dyn1)
            log_progress(task_id, "[完成] Trend Layer 提取完成")

        # ---- Volatility Layer ----
        log_progress(task_id, "=" * 50, 'divider')
        log_progress(task_id, "PHASE 1.5b: 波动率分析 (Volatility Layer)", 'phase')
        log_progress(task_id, "=" * 50, 'divider')

        log_progress(task_id, "[Vol] 计算滚动波动率...")
        vol1 = volatility.compute_volatility_layer(dyn1, rolling_window=10, fit_garch=True)
        results['volatility'] = vol1
        results['volatility_summary'] = volatility.summarize_volatility(vol1)
        vol_summ = results['volatility_summary']

        log_progress(task_id, "[完成] Volatility Layer:")
        for key in ["energy", "brightness", "complexity", "rhythm"]:
            s = vol_summ[key]
            log_progress(task_id,
                f"  {key}: mean_vol={s['mean_vol']:.5f}, regime={s['volatility_regime']}, "
                f"GARCH α+β={s.get('garch_persistence') or 0:.3f} "
                f"(converged={s.get('garch_converged', False)})")

        # ---- Trend predictions (fast models only: ARIMA + HMM) ----
        log_progress(task_id, "[TrendPred] 对4个趋势进行预测 (ARIMA + HMM)...")
        trend_preds = prediction.predict_all_trends(
            dyn1, forecast_horizon=forecast_horizon, models="ARIMA,HMM", verbose=False
        )
        results['trend_predictions'] = trend_preds
        log_progress(task_id, "[完成] 趋势预测完成")

        # ---- Volatility predictions (fast models only) ----
        log_progress(task_id, "[VolPred] 对波动率进行预测 (ARIMA + HMM)...")
        vol_preds = prediction.predict_all_volatilities(
            vol1, forecast_horizon=min(10, forecast_horizon), models="ARIMA,HMM", verbose=False
        )
        results['volatility_predictions'] = vol_preds
        log_progress(task_id, "[完成] 波动率预测完成")

        if dual_audio and filepath2:
            if '_y2' not in results:
                y2, _ = loader.load_audio(filepath2, target_sr=sr)
                results['_y2'] = y2

        current_step += 1
        update_progress(task_id, int(current_step / total_steps * 100))

    # ============================================================
    # Model Ensemble Structural Analysis
    # ============================================================
    if 'model_analysis' in analysis_options:
        # Ensure dynamics are extracted
        if dyn1 is None:
            log_progress(task_id, "[Dyn] 先提取动态特征...")
            dyn1 = dynamics.extract_dynamics(y1, sr, window_size=0.5, hop_size=0.25)
            results['dynamics'] = dyn1

        log_progress(task_id, "=" * 50, 'divider')
        log_progress(task_id, "PHASE 2a: 模型结构分析 (ARIMA/HMM/LSTM/Transformer)", 'phase')
        log_progress(task_id, "=" * 50, 'divider')

        log_progress(task_id, "[Model] 运行四种模型的结构侦探分析...")
        model_report = model_analysis.analyze_model_ensemble(
            dyn1, n_hmm_states=3, lstm_epochs=15, transformer_epochs=15, verbose=False)
        results['model_analysis'] = model_report

        # Summarize key findings
        if model_report.arima:
            log_progress(task_id, f"  ARIMA: {model_report.arima.summary}")
        if model_report.hmm:
            log_progress(task_id, f"  HMM:   {model_report.hmm.n_states} 状态, 区分度 {model_report.hmm.segmentation_quality:.0%}")
        if model_report.lstm:
            log_progress(task_id, f"  LSTM:  最优记忆 {model_report.lstm.optimal_lookback_seconds:.1f}s, 最可学: {model_report.lstm.most_learnable}")
        if model_report.transformer:
            log_progress(task_id, f"  Transformer: {model_report.transformer.n_distinct_layers} 个时间尺度")

        current_step += 1
        update_progress(task_id, int(current_step / total_steps * 100))

    # ============================================================
    # Time Series Analysis
    # ============================================================
    if 'timeseries' in analysis_options:
        log_progress(task_id, "=" * 50, 'divider')
        log_progress(task_id, "PHASE 2: 时序分析", 'phase')
        log_progress(task_id, "=" * 50, 'divider')
        
        y_for_ts = results.get('waveform', {}).get('y', y1)
        
        log_progress(task_id, "[2.1] 计算ACF & PACF...")
        lags, acf_vals, ci = analysis.compute_acf(y_for_ts[:min(len(y_for_ts), 4000)], nlags=40)
        _, pacf_vals, _ = analysis.compute_pacf(y_for_ts[:min(len(y_for_ts), 4000)], nlags=40)
        results['acf_pacf'] = {'lags': lags, 'acf': acf_vals, 'pacf': pacf_vals, 'ci': ci}
        
        log_progress(task_id, "[2.2] 周期性分析...")
        periodicity = analysis.analyze_periodicity(y1, sr)
        results['periodicity'] = periodicity
        log_progress(task_id, f"      主频率: {periodicity['dominant_frequency']:.1f} Hz")
        log_progress(task_id, f"      主周期: {periodicity['dominant_period_seconds']:.4f}s")
        
        log_progress(task_id, "[2.3] 复杂度分析...")
        complexity = analysis.analyze_complexity(y1)
        results['complexity'] = complexity
        log_progress(task_id, f"      过零率: {complexity['zero_crossing_rate']:.4f}")
        log_progress(task_id, f"      样本熵: {complexity['sample_entropy']:.4f}")
        
        log_progress(task_id, "[2.4] 频谱平坦度...")
        fft_mag_for_sf = results.get('fft', {}).get('mag', features.compute_fft(y1, sr)[1])
        flatness = analysis.compute_spectral_flatness(fft_mag_for_sf)
        results['spectral_flatness'] = flatness
        log_progress(task_id, f"      频谱平坦度: {flatness:.4f} (0=音调, 1=噪声)")
        
        current_step += 1
        update_progress(task_id, int(current_step / total_steps * 100))
    
    # ============================================================
    # Unsupervised Pattern Discovery
    # ============================================================
    if 'unsupervised' in analysis_options:
        log_progress(task_id, "=" * 50, 'divider')
        log_progress(task_id, "PHASE 3: 无监督模式发现", 'phase')
        log_progress(task_id, "=" * 50, 'divider')
        
        log_progress(task_id, "[Unsup] 运行聚类和motif检测...")
        unsup_report = unsupervised.explore_unsupervised(
            y1, sr, n_components=4, n_clusters=4, verbose=False,
        )
        results['unsupervised'] = unsup_report
        log_progress(task_id, f"[完成] 发现 {len(unsup_report.change_points)} 个变点, {len(unsup_report.motifs)} 个motif")
        
        current_step += 1
        update_progress(task_id, int(current_step / total_steps * 100))

    # ============================================================
    # Prediction & Band Analysis
    # ============================================================
    if 'prediction' in analysis_options or 'band' in analysis_options:
        if 'features' in analysis_options and 'mel' in results:
            mel_spec = results['mel']['spec']
        else:
            _, _, mel_spec = features.compute_mel_spectrogram(y1, sr, n_mels=n_mels)
        
        if 'prediction' in analysis_options:
            log_progress(task_id, "=" * 50, 'divider')
            log_progress(task_id, "PHASE 4: 机器学习预测", 'phase')
            log_progress(task_id, "=" * 50, 'divider')
            
            log_progress(task_id, "[Pred] 运行所有预测模型 (ARIMA, HMM, LSTM, Transformer)...")
            results['predictions'] = prediction.run_all_predictions(
                mel_spec, forecast_horizon=forecast_horizon, epochs=30, verbose=False
            )
            
            log_progress(task_id, "[完成] 模型性能摘要:")
            for model_name, (forecast, metrics, true) in results['predictions'].items():
                if metrics:
                    log_progress(task_id, f"  {model_name}: RMSE={metrics.get('RMSE', 'N/A'):.4f}, MAE={metrics.get('MAE', 'N/A'):.4f}")
            
            current_step += 1
            update_progress(task_id, int(current_step / total_steps * 100))
        
        if 'band' in analysis_options:
            log_progress(task_id, "=" * 50, 'divider')
            log_progress(task_id, "PHASE 4.5: 频带分析", 'phase')
            log_progress(task_id, "=" * 50, 'divider')
            
            log_progress(task_id, "[Band] 分析频带可预测性...")
            band_results = band_analysis.analyze_band_predictability(
                mel_spec, forecast_horizon=forecast_horizon, epochs=30
            )
            results['band_results'] = band_results
            results['band_summary'] = band_analysis.compute_band_error_summary(band_results)
            results['predictability_rank'] = band_analysis.get_predictability_rank(results['band_summary'])
            
            log_progress(task_id, "[完成] 频带可预测性排名:")
            for i, item in enumerate(results['predictability_rank'], 1):
                log_progress(task_id, f"  {i}. {item['band']} - 最佳: {item['best_model']}, RMSE: {item['avg_rmse']:.4f}")
            
            current_step += 1
            update_progress(task_id, int(current_step / total_steps * 100))
    
    # ============================================================
    # Dual Audio Analysis
    # ============================================================
    if 'comparison' in analysis_options and dual_audio and filepath2:
        log_progress(task_id, "=" * 50, 'divider')
        log_progress(task_id, "PHASE 5: 双音频对比", 'phase')
        log_progress(task_id, "=" * 50, 'divider')
        
        log_progress(task_id, f"[Comp] 加载第二个音频: {os.path.basename(filepath2)}")
        y2, sr2 = loader.load_audio(filepath2, target_sr=sr)
        results['_y2'] = y2

        log_progress(task_id, "[Comp] 运行多维探索分析...")
        disc_report = discovery.explore(
            y1, sr, y2, sr2,
            window_size=0.5, hop_size=0.25, verbose=False,
        )
        results['discovery'] = disc_report
        log_progress(task_id, f"[完成] 发现 {len(disc_report.discoveries)} 个匹配, {len(disc_report.contrasts)} 个对比")

        # ---- Volatility similarity (if dynamics_analysis was also run) ----
        if 'dynamics_analysis' in analysis_options and vol1 is not None:
            log_progress(task_id, "[Comp] 计算波动率相似度...")
            # Ensure we have vol2
            if dyn1 is not None:
                dyn2_comp = dynamics.extract_dynamics(y2, sr, window_size=0.5, hop_size=0.25)
            else:
                dyn2_comp = dynamics.extract_dynamics(y2, sr, window_size=0.5, hop_size=0.25)
            vol2_comp = volatility.compute_volatility_layer(dyn2_comp, rolling_window=10, fit_garch=True)
            results['_vol2'] = vol2_comp

            vol_sim_result = volatility.compute_volatility_similarity(vol1, vol2_comp)
            results['volatility_similarity'] = vol_sim_result
            log_progress(task_id,
                f"[完成] 波动率相似度: {vol_sim_result['global_volatility_similarity']:.1f}% "
                f"(主导维度: {vol_sim_result['dominant_trend']})")

            # Also compute dynamics similarity
            log_progress(task_id, "[Comp] 计算动态趋势相似度...")
            dyn_sim = dynamics.compute_dynamics_similarity(dyn1, dyn2_comp)
            results['dynamics_similarity'] = dyn_sim
            log_progress(task_id,
                f"[完成] 动态趋势相似度: {dyn_sim['global_dynamics_similarity']:.1f}%")

        current_step += 1
        update_progress(task_id, int(current_step / total_steps * 100))

    # Store analysis options used
    results['analysis_options'] = analysis_options

    return results


@app.route('/', methods=['GET', 'POST'])
def index():
    """Main page with upload form."""
    if request.method == 'POST':
        # Generate unique ID for this analysis
        task_id = str(uuid.uuid4())[:8]
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], task_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize progress
        analysis_progress[task_id] = {'logs': [], 'status': 'running', 'progress': 0}
        
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
        
        # Check if we have dual audio
        has_dual_audio = filepath2 is not None
        
        # Use synthetic audio if no file uploaded
        if not filepath1:
            y1, sr = loader.generate_sample_audio(duration=5.0)
            filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_synthetic.wav")
            import soundfile as sf
            sf.write(filepath1, y1, sr)
        
        # Get parameters
        forecast_horizon = int(request.form.get('forecast_horizon', 20))
        n_mels = int(request.form.get('n_mels', 128))
        
        # Parse analysis options
        analysis_options = parse_analysis_options(request.form, has_dual_audio)
        
        # Store task info for async processing
        task_info = {
            'task_id': task_id,
            'experiment_name': request.form.get('experiment_name', '').strip() or None,
            'filepath1': filepath1,
            'filepath1_name': os.path.basename(filepath1),
            'filepath2': filepath2,
            'filepath2_name': os.path.basename(filepath2) if filepath2 else None,
            'forecast_horizon': forecast_horizon,
            'n_mels': n_mels,
            'analysis_options': analysis_options,
            'output_dir': output_dir
        }
        
        # Save task info
        with open(os.path.join(output_dir, 'task_info.json'), 'w') as f:
            json.dump({k: v for k, v in task_info.items() if k != 'results'}, f, default=str)
        
        # Return task_id for SSE streaming
        return render_template('analysis.html', task_id=task_id, analysis_options=analysis_options)
    
    return render_template('index.html', analysis_options=ANALYSIS_OPTIONS)


@app.route('/stream/<task_id>')
def stream(task_id):
    """SSE endpoint for streaming analysis progress."""
    def generate():
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], task_id)
        
        # Load task info
        try:
            with open(os.path.join(output_dir, 'task_info.json'), 'r') as f:
                task_info = json.load(f)
        except:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Task not found'})}\n\n"
            return
        
        # Send initial status
        yield f"data: {json.dumps({'type': 'status', 'status': 'running', 'progress': 0})}\n\n"
        
        # Run analysis in a separate thread
        def run_analysis():
            try:
                stdout_logger = ProgressWriter(task_id)
                stderr_logger = ProgressWriter(task_id, level='error')
                with redirect_stdout(stdout_logger), redirect_stderr(stderr_logger):
                    results = analyze_audio_with_progress(
                        task_id,
                        task_info['filepath1'],
                        task_info.get('filepath2'),
                        task_info['forecast_horizon'],
                        task_info['n_mels'],
                        task_info['analysis_options']
                    )
                
                # Generate plots if visualization is enabled
                plot_files = []
                if 'visualization' in task_info['analysis_options']:
                    log_progress(task_id, "=" * 50, 'divider')
                    log_progress(task_id, "PHASE 6: 生成可视化", 'phase')
                    log_progress(task_id, "=" * 50, 'divider')
                    
                    log_progress(task_id, "[Viz] 生成分析图表...")
                    plot_files = visualization.generate_report_plots(results, output_dir)

                    # Generate discovery plots
                    if results.get('discovery'):
                        disc = results['discovery']
                        dyn_a = results.get('dynamics')
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

                    # Generate dynamics plots
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

                    # Generate volatility plots
                    if results.get('volatility') and results.get('dynamics'):
                        log_progress(task_id, "[Viz] 生成波动率分析图表...")
                        vol_dir = os.path.join(output_dir, 'volatility')
                        os.makedirs(vol_dir, exist_ok=True)
                        vol = results['volatility']
                        dyn = results['dynamics']

                        visualization.plot_volatility_layer(
                            dyn, vol,
                            save_path=os.path.join(vol_dir, 'volatility_layer.png'),
                        )
                        plot_files.append('volatility/volatility_layer.png')

                        # GARCH diagnostics for energy
                        visualization.plot_garch_diagnostics(
                            vol, trend_key='energy',
                            save_path=os.path.join(vol_dir, 'garch_energy.png'),
                        )
                        plot_files.append('volatility/garch_energy.png')

                        # Dynamics analysis summary dashboard
                        dyn_analysis = {
                            'trend_summary': dynamics.summarize_dynamics(dyn),
                            'volatility_summary': results.get('volatility_summary', {}),
                        }
                        visualization.plot_dynamics_analysis_summary(
                            dyn_analysis,
                            save_path=os.path.join(vol_dir, 'dynamics_analysis_summary.png'),
                        )
                        plot_files.append('volatility/dynamics_analysis_summary.png')

                        # Volatility comparison (if dual audio)
                        vol2_viz = results.get('_vol2')
                        if vol2_viz is not None:
                            vol_sim = results.get('volatility_similarity')
                            visualization.plot_volatility_comparison(
                                vol, vol2_viz, sim_result=vol_sim,
                                save_path=os.path.join(vol_dir, 'volatility_comparison.png'),
                            )
                            plot_files.append('volatility/volatility_comparison.png')

                    log_progress(task_id, f"[完成] 生成了 {len(plot_files)} 个图表")

                # Create serializable versions
                if results.get('discovery'):
                    disc = results['discovery']
                    def _profile_to_dict(p):
                        if p is None: return None
                        return {'rhythm_signature': p.rhythm_signature, 'energy_profile': p.energy_profile,
                                'timbre_quality': p.timbre_quality, 'standout_features': p.standout_features}
                    def _discovery_to_dict(d):
                        return {'title': d.title, 'dimension': d.dimension, 'discovery_type': d.discovery_type,
                                'summary': d.summary, 'n_matches': len(d.segment_matches),
                                'avg_confidence': d.meta.get('avg_confidence', 0)}
                    results['discovery_serializable'] = {
                        'audio_a_profile': _profile_to_dict(disc.audio_a_profile),
                        'audio_b_profile': _profile_to_dict(disc.audio_b_profile),
                        'n_discoveries': len(disc.discoveries), 'n_contrasts': len(disc.contrasts),
                        'overview': disc.overview, 'params': disc.params,
                    }

                if results.get('dynamics'):
                    from audiots.dynamics import summarize_dynamics
                    dyn_summary = summarize_dynamics(results['dynamics'])
                    seg = results.get('dynamics_segments', {})
                    results['dynamics_serializable'] = {
                        'summary': dyn_summary,
                        'n_climax': len(seg.get('climax_indices', [])),
                        'n_calm': len(seg.get('calm_indices', [])),
                    }

                if results.get('volatility'):
                    from audiots.volatility import summarize_volatility
                    vol_summary = results.get('volatility_summary', summarize_volatility(results['volatility']))
                    results['volatility_serializable'] = {
                        'summary': {
                            k: {sk: sv for sk, sv in v.items()
                                if not isinstance(sv, np.ndarray)}
                            for k, v in vol_summary.items()
                        },
                        'global_vol_similarity': (
                            results.get('volatility_similarity', {}).get('global_volatility_similarity')
                        ),
                    }

                if results.get('trend_predictions'):
                    # Simplify trend predictions for JSON
                    trend_pred_serializable = {}
                    for trend_key, preds in results['trend_predictions'].items():
                        trend_pred_serializable[trend_key] = {}
                        for model_name, (forecast, metrics, true) in preds.items():
                            trend_pred_serializable[trend_key][model_name] = {
                                'rmse': float(metrics.get('RMSE', np.nan)) if not np.isnan(metrics.get('RMSE', np.nan)) else None,
                                'mae': float(metrics.get('MAE', np.nan)) if not np.isnan(metrics.get('MAE', np.nan)) else None,
                            }
                    results['trend_predictions_serializable'] = trend_pred_serializable

                if results.get('unsupervised'):
                    u = results['unsupervised']
                    results['unsupervised_serializable'] = {
                        'n_change_points': len(u.change_points),
                        'n_segments': len(u.segments),
                        'n_motifs': len(u.motifs),
                        'overview': u.overview,
                    }

                # Store results
                results['task_id'] = task_id
                results['plot_files'] = plot_files
                results['params'] = {
                    'experiment_name': task_info.get('experiment_name'),
                    'forecast_horizon': task_info['forecast_horizon'],
                    'n_mels': task_info['n_mels'],
                    'audio1_name': os.path.basename(task_info['filepath1']),
                    'audio2_name': os.path.basename(task_info['filepath2']) if task_info.get('filepath2') else None,
                    'analysis_options': task_info['analysis_options']
                }
                results['experiment_name'] = task_info.get('experiment_name') or ''
                
                # Save results
                serializable = {k: v for k, v in results.items()
                                if not k.startswith('_') and k not in (
                                    'similarity', 'discovery', 'unsupervised',
                                    'dynamics', 'dynamics_2',
                                    'dynamics_segments', 'dynamics_segments_2',
                                    'dynamics_similarity',
                                    'volatility', 'volatility_summary',
                                    'trend_predictions', 'volatility_predictions',
                                    'band_results',  # Will be processed separately
                                )}
                
                # Process band_results for serialization
                if results.get('band_results'):
                    band_results_serializable = {}
                    for band_key, band_data in results['band_results'].items():
                        band_results_serializable[band_key] = {
                            'info': band_data['info'],
                            'predictions': {}
                        }
                        for model_name, model_data in band_data['predictions'].items():
                            band_results_serializable[band_key]['predictions'][model_name] = {
                                'metrics': model_data['metrics']
                            }
                    serializable['band_results'] = band_results_serializable
                
                if results.get('discovery_serializable'):
                    serializable['discovery'] = results['discovery_serializable']
                if results.get('unsupervised_serializable'):
                    serializable['unsupervised'] = results['unsupervised_serializable']
                if results.get('dynamics_serializable'):
                    serializable['dynamics'] = results['dynamics_serializable']
                if results.get('volatility_serializable'):
                    serializable['volatility'] = results['volatility_serializable']
                if results.get('trend_predictions_serializable'):
                    serializable['trend_predictions'] = results['trend_predictions_serializable']

                if results.get('model_analysis'):
                    mr = results['model_analysis']
                    serializable['model_analysis'] = {
                        'arima_summary': mr.arima.summary if mr.arima else None,
                        'hmm_summary': mr.hmm.summary if mr.hmm else None,
                        'hmm_n_states': mr.hmm.n_states if mr.hmm else 0,
                        'hmm_state_profiles': [
                            {'id': p.state_id, 'label': p.label, 'fraction': p.fraction,
                             'description': p.description}
                            for p in (mr.hmm.state_profiles if mr.hmm else [])
                        ],
                        'lstm_summary': mr.lstm.summary if mr.lstm else None,
                        'lstm_optimal_lookback_s': mr.lstm.optimal_lookback_seconds if mr.lstm else None,
                        'lstm_most_learnable': mr.lstm.most_learnable if mr.lstm else None,
                        'transformer_summary': mr.transformer.summary if mr.transformer else None,
                        'transformer_n_layers': mr.transformer.n_distinct_layers if mr.transformer else 0,
                        'ensemble_summary': mr.ensemble_summary,
                    }
                if results.get('volatility_similarity'):
                    serializable['volatility_similarity'] = results['volatility_similarity']
                if results.get('dynamics_similarity'):
                    serializable['dynamics_similarity'] = {
                        'global_dynamics_similarity': results['dynamics_similarity']['global_dynamics_similarity'],
                        'dominant_trend': results['dynamics_similarity']['dominant_trend'],
                        'structural_coherence': results['dynamics_similarity']['structural_coherence'],
                    }

                # Add analysis log
                if task_id in analysis_progress and analysis_progress[task_id].get('logs'):
                    serializable['analysis_log'] = [
                        f"[{log['time']}] {log['message']}"
                        for log in analysis_progress[task_id]['logs']
                    ]
                serializable['experiment_name'] = task_info.get('experiment_name') or ''

                with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as f:
                    json.dump(serializable, f, default=str, ensure_ascii=False)
                
                log_progress(task_id, "=" * 50, 'divider')
                log_progress(task_id, "Analysis completed!", 'success')
                log_progress(task_id, "=" * 50, 'divider')
                update_progress(task_id, 100, 'completed')
                
            except Exception as e:
                log_progress(task_id, f"错误: {str(e)}", 'error')
                update_progress(task_id, 0, 'error')
        
        # Start analysis thread
        import threading
        thread = threading.Thread(target=run_analysis)
        thread.start()
        
        # Stream logs
        last_log_index = 0
        while True:
            if task_id in analysis_progress:
                progress_data = analysis_progress[task_id]
                
                # Send new logs
                logs = progress_data['logs']
                for i in range(last_log_index, len(logs)):
                    yield f"data: {json.dumps({'type': 'log', 'log': logs[i]})}\n\n"
                last_log_index = len(logs)
                
                # Send progress update
                yield f"data: {json.dumps({'type': 'progress', 'progress': progress_data['progress'], 'status': progress_data['status']})}\n\n"
                
                # Check if completed
                if progress_data['status'] in ['completed', 'error']:
                    if progress_data['status'] == 'completed':
                        yield f"data: {json.dumps({'type': 'complete', 'task_id': task_id})}\n\n"
                    break
            
            time.sleep(0.1)
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@app.route('/results/history')
def results_history():
    """Return JSON list of historical result tasks."""
    output_root = app.config['OUTPUT_FOLDER']
    history = []

    if os.path.exists(output_root):
        task_dirs = []
        for task_id in os.listdir(output_root):
            output_dir = os.path.join(output_root, task_id)
            if not os.path.isdir(output_dir):
                continue
            task_dirs.append((task_id, os.path.getmtime(output_dir)))

        for task_id, _ in sorted(task_dirs, key=lambda item: item[1], reverse=True):
            output_dir = os.path.join(output_root, task_id)
            results_file = os.path.join(output_dir, 'results.json')
            if not os.path.exists(results_file):
                continue

            entry = {'task_id': task_id}
            task_info_file = os.path.join(output_dir, 'task_info.json')
            if os.path.exists(task_info_file):
                try:
                    with open(task_info_file, 'r', encoding='utf-8') as f:
                        info = json.load(f)
                    entry['experiment_name'] = info.get('experiment_name') or None
                    entry['audio1_name'] = info.get('audio1_name', info.get('filepath1_name', 'Unknown'))
                    entry['analysis_options'] = info.get('analysis_options', [])
                    entry['forecast_horizon'] = info.get('forecast_horizon')
                except Exception:
                    entry['experiment_name'] = None
                    entry['audio1_name'] = 'Unknown'
                    entry['analysis_options'] = []
                    entry['forecast_horizon'] = None
            else:
                entry['experiment_name'] = None
                entry['audio1_name'] = 'Unknown'
                entry['analysis_options'] = []
                entry['forecast_horizon'] = None

            entry['created_at'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(output_dir)))
            history.append(entry)

    return jsonify(history)


@app.route('/results/<task_id>')
def results(task_id):
    """Display analysis results."""
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], task_id)
    
    if not os.path.exists(output_dir):
        return "分析任务不存在", 404
    
    # Load results
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


def secure_filename(filename):
    """Simple secure filename function."""
    import re
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', filename)


# ============================================
# Experiment Naming API
# ============================================
EXPERIMENTS_FILE = 'experiments.json'

def load_experiments():
    """Load experiments from JSON file."""
    if os.path.exists(EXPERIMENTS_FILE):
        try:
            with open(EXPERIMENTS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def save_experiments(experiments):
    """Save experiments to JSON file."""
    with open(EXPERIMENTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(experiments, f, ensure_ascii=False, indent=2)

@app.route('/api/experiments')
def get_experiments():
    """Get all saved experiments."""
    experiments = load_experiments()
    return jsonify(experiments)

@app.route('/api/experiments', methods=['POST'])
def save_experiment():
    """Save an experiment with a name."""
    data = request.get_json()
    name = data.get('name', '').strip()
    task_id = data.get('task_id', '').strip()

    if not name or not task_id:
        return jsonify({'error': 'Name and task_id are required'}), 400

    experiments = load_experiments()

    # Check if experiment with same name exists, update it
    for exp in experiments:
        if exp['name'] == name:
            exp['task_id'] = task_id
            exp['updated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
            save_experiments(experiments)
            return jsonify({'success': True, 'message': 'Experiment updated'})

    # Add new experiment
    experiments.append({
        'name': name,
        'task_id': task_id,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'updated_at': time.strftime('%Y-%m-%d %H:%M:%S')
    })
    save_experiments(experiments)
    return jsonify({'success': True, 'message': 'Experiment saved'})

@app.route('/api/experiments/<name>', methods=['DELETE'])
def delete_experiment(name):
    """Delete an experiment by name."""
    experiments = load_experiments()
    original_len = len(experiments)
    experiments = [e for e in experiments if e['name'] != name]

    if len(experiments) == original_len:
        return jsonify({'error': 'Experiment not found'}), 404

    save_experiments(experiments)
    return jsonify({'success': True, 'message': 'Experiment deleted'})

@app.route('/api/experiments/<name>')
def get_experiment(name):
    """Get an experiment by name."""
    experiments = load_experiments()
    for exp in experiments:
        if exp['name'] == name:
            return jsonify(exp)
    return jsonify({'error': 'Experiment not found'}), 404


@app.route('/docs')
@app.route('/docs/')
@app.route('/docs/<page>')
def docs(page=None):
    """Serve documentation pages (rendered from Markdown)."""
    import markdown

    docs_dir = os.path.join(os.path.dirname(__file__), 'docs')

    # Map page names to files
    doc_files = {
        'guide': 'guide.md',
        'theory': 'theory.md',
        'analysis': 'analysis.md',
        'models': 'models.md',
        'results': 'results.md',
        'faq': 'faq.md',
        'genre': 'genre.md',
    }

    if page is None:
        # Docs index page
        md_content = """# Audio Lab 文档中心

欢迎阅读 **Audio Lab** 音频时间序列分析平台文档。本平台集成了经典时间序列分析方法与现代深度学习技术，为音频信号分析提供全方位解决方案。

## 📚 文档目录

| 文档 | 内容概述 |
|------|----------|
| [**使用指南**](/docs/guide) | Web 界面操作说明、命令行参数、各分析选项详解 |
| [**理论基础**](/docs/theory) | 从声波振动到数字信号处理的教学讲解，包含数学公式推导 |
| [**分析模块详解**](/docs/analysis) | 每个分析模块的算法原理、输入输出、实现细节 |
| [**深度学习模型**](/docs/models) | LSTM、Transformer等深度学习模型的架构、训练策略、数据划分 |
| [**结果解读**](/docs/results) | 如何看懂图表、统计指标、白噪声检验结果 |
| [**常见问题 FAQ**](/docs/faq) | 常见问题解答、故障排除、最佳实践 |
| [**音乐风格分析**](/docs/genre) | GTZAN数据集音乐风格分类分析报告（预留） |

## 🎯 平台特色

- **多模型集成**：ARIMA、HMM、LSTM、Transformer 四大模型协同预测
- **频带分析**：按频率带分析可预测性，揭示音频内在结构
- **动态特征**：能量、亮度、复杂度、节奏等多维度动态分析
- **波动率建模**：GARCH模型捕捉音频波动特性
- **白噪声检验**：6种统计检验综合判断信号随机性
- **可视化报告**：自动生成高质量分析图表

---

返回 [主页](/) | [项目 GitHub](https://github.com/OPSAF/Audio-TSA)
"""
        html = markdown.markdown(
            md_content,
            extensions=[
                'markdown.extensions.tables',
                'markdown.extensions.fenced_code',
                'markdown.extensions.codehilite'
            ],
            output_format='html5'
        )
        if request.args.get('ajax') == '1':
            return jsonify({'title': '文档', 'content': html, 'page': 'index'})
        return render_template('docs.html', content=html, title='文档', page='index')

    if page not in doc_files:
        return "文档页面不存在", 404

    filepath = os.path.join(docs_dir, doc_files[page])
    if not os.path.exists(filepath):
        return "文档文件不存在", 404

    with open(filepath, 'r', encoding='utf-8') as f:
        md_content = f.read()

    html = markdown.markdown(
        md_content,
        extensions=[
            'markdown.extensions.tables',
            'markdown.extensions.fenced_code',
            'markdown.extensions.codehilite'
        ],
        output_format='html5'
    )

    titles = {
        'guide': '使用指南',
        'theory': '理论基础',
        'analysis': '分析模块详解',
        'models': '深度学习模型',
        'results': '结果解读',
        'faq': '常见问题 FAQ',
        'genre': '音乐风格分析',
    }

    title = titles.get(page, page)
    if request.args.get('ajax') == '1':
        return jsonify({'title': title, 'content': html, 'page': page})

    return render_template('docs.html', content=html, title=title, page=page)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)