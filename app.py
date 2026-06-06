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
from audiots import loader
from audiots.pipeline import run_full_analysis, generate_all_plots, serialize_results

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
    """Perform complete audio analysis — delegates to shared pipeline."""
    if analysis_options is None:
        analysis_options = [opt['id'] for opt in ANALYSIS_OPTIONS if opt['default']]

    dual_audio = filepath2 is not None

    # ── Load audio 1 ────────────────────────────────────────────────────
    log_progress(task_id, f"加载音频文件: {os.path.basename(filepath1)}")
    y1, sr = loader.load_audio(filepath1, target_sr=16000)
    log_progress(task_id, f"音频信息: 时长={len(y1)/sr:.2f}s, 采样率={sr}Hz, 样本数={len(y1)}")
    update_progress(task_id, 5)

    # ── Load audio 2 (if dual) ───────────────────────────────────────────
    y2, sr2 = None, None
    if dual_audio and filepath2:
        y2, sr2 = loader.load_audio(filepath2, target_sr=sr)

    # ── Progress callback ────────────────────────────────────────────────
    def pipe_log(msg: str, level: str = 'info'):
        log_progress(task_id, msg, level=level)

    # ── Delegate to shared pipeline ──────────────────────────────────────
    results = run_full_analysis(
        y1=y1, sr=sr, y2=y2, sr2=sr2,
        forecast_horizon=forecast_horizon,
        n_mels=n_mels,
        analysis_options=analysis_options,
        progress_callback=pipe_log,
        fast=False,
    )

    # Store y2 for viz
    if y2 is not None:
        results['_y2'] = y2

    update_progress(task_id, 85)
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

                # ── Visualization (shared pipeline) ────────────────────
                plot_files = []
                if 'visualization' in task_info['analysis_options']:
                    log_progress(task_id, "=" * 50, 'divider')
                    log_progress(task_id, "PHASE 6: 生成可视化", 'phase')
                    log_progress(task_id, "=" * 50, 'divider')
                    try:
                        plot_files = generate_all_plots(
                            results, output_dir,
                            y1=results.get('_y1'),
                            sr=results.get('_sr', 16000),
                            y2=results.get('_y2'),
                        )
                    except Exception as e:
                        log_progress(task_id, f"[警告] 可视化生成失败: {str(e)}", 'warning')

                # ── Serialization (shared pipeline) ────────────────────
                results['task_id'] = task_id
                serializable = serialize_results(
                    results, task_id=task_id,
                    task_info={
                        'experiment_name': task_info.get('experiment_name'),
                        'forecast_horizon': task_info['forecast_horizon'],
                        'n_mels': task_info['n_mels'],
                        'audio1_name': os.path.basename(task_info['filepath1']),
                        'audio2_name': os.path.basename(task_info['filepath2']) if task_info.get('filepath2') else None,
                        'analysis_options': task_info['analysis_options'],
                    },
                    plot_files=plot_files,
                )
                serializable['experiment_name'] = task_info.get('experiment_name') or ''

                # Attach analysis log
                if task_id in analysis_progress and analysis_progress[task_id].get('logs'):
                    serializable['analysis_log'] = [
                        f"[{log['time']}] {log['message']}"
                        for log in analysis_progress[task_id]['logs']
                    ]

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
@app.route('/docs/<path:page>')
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
        'batch_analyze': 'batch_analyze.md',
        'genre': 'genre.md',
        # Genre sub-pages
        'genre/rock': 'genre/rock.md',
        'genre/reggae': 'genre/reggae.md',
        'genre/pop': 'genre/pop.md',
        'genre/metal': 'genre/metal.md',
        'genre/jazz': 'genre/jazz.md',
        'genre/hiphop': 'genre/hiphop.md',
        'genre/disco': 'genre/disco.md',
        'genre/country': 'genre/country.md',
        'genre/classical': 'genre/classical.md',
        'genre/blues': 'genre/blues.md',
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
        'batch_analyze': '批量分析工具',
        'genre': '音乐风格分析',
        'genre/rock': '摇滚 (Rock) 分析',
        'genre/reggae': '雷鬼 (Reggae) 分析',
        'genre/pop': '流行 (Pop) 分析',
        'genre/metal': '金属 (Metal) 分析',
        'genre/jazz': '爵士 (Jazz) 分析',
        'genre/hiphop': '嘻哈 (Hip-hop) 分析',
        'genre/disco': '迪斯科 (Disco) 分析',
        'genre/country': '乡村 (Country) 分析',
        'genre/classical': '古典 (Classical) 分析',
        'genre/blues': '蓝调 (Blues) 分析',
    }

    title = titles.get(page, page)
    if request.args.get('ajax') == '1':
        return jsonify({'title': title, 'content': html, 'page': page})

    return render_template('docs.html', content=html, title=title, page=page)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)