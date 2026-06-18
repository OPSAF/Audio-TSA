"""Build a self-contained HTML file with all markdown reports embedded."""
import os

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
GENRES_DIR = os.path.join(DOCS_DIR, 'genres')

# ── Read all markdown files ──
pages = {}
pages['index'] = open(os.path.join(DOCS_DIR, 'index.md'), 'r', encoding='utf-8').read()

for f in sorted(os.listdir(GENRES_DIR)):
    if f.endswith('.md'):
        name = 'genres/' + f.replace('.md', '')
        pages[name] = open(os.path.join(GENRES_DIR, f), 'r', encoding='utf-8').read()

# ── Build JS pages object ──
lines = ['const PAGES = {']
for name, content in pages.items():
    # Escape backslashes, backticks, and dollar signs for JS template literal
    escaped = content.replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$')
    lines.append(f"  '{name}': `{escaped}`,")
lines.append('};')
pages_js = '\n'.join(lines)

# ── HTML template ──
html = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🎵 音乐流派音频分析报告</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #c9d1d9; --text-secondary: #8b949e; --accent: #58a6ff;
    --accent2: #f78166; --accent3: #3fb950; --accent4: #d2a8ff;
    --code-bg: #1c2128; --table-stripe: #1a2029; --red: #f85149;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans SC', sans-serif;
    background: var(--bg); color: var(--text); display: flex; min-height: 100vh; line-height: 1.7;
  }
  nav {
    width: 280px; min-width: 280px; background: var(--surface);
    border-right: 1px solid var(--border); padding: 24px 0;
    position: sticky; top: 0; height: 100vh; overflow-y: auto; z-index: 10;
  }
  nav .logo { font-size: 1.3em; font-weight: 700; padding: 0 20px 20px; border-bottom: 1px solid var(--border); margin-bottom: 16px; color: var(--text); }
  nav .logo span { color: var(--accent2); }
  nav a { display: block; padding: 8px 20px; color: var(--text-secondary); text-decoration: none; font-size: 0.92em; transition: all 0.15s; border-left: 3px solid transparent; margin: 2px 0; cursor: pointer; }
  nav a:hover { color: var(--text); background: rgba(88,166,255,0.06); border-left-color: var(--accent); }
  nav a.active { color: var(--text); background: rgba(88,166,255,0.1); border-left-color: var(--accent); font-weight: 600; }
  nav .section-title { font-size: 0.75em; text-transform: uppercase; letter-spacing: 1px; color: var(--text-secondary); padding: 16px 20px 6px; opacity: 0.7; }
  main { flex: 1; padding: 40px 48px; max-width: 960px; overflow-x: hidden; }
  .markdown-body h1 { font-size: 2em; margin: 0 0 8px; padding-bottom: 12px; border-bottom: 2px solid var(--border); color: #fff; }
  .markdown-body h2 { font-size: 1.4em; margin: 36px 0 12px; padding-bottom: 6px; border-bottom: 1px solid var(--border); color: var(--accent); }
  .markdown-body h3 { font-size: 1.1em; margin: 24px 0 8px; color: var(--accent4); }
  .markdown-body p { margin: 10px 0; }
  .markdown-body blockquote { border-left: 3px solid var(--accent2); padding: 8px 16px; margin: 16px 0; background: rgba(247,129,102,0.06); color: var(--text-secondary); border-radius: 0 6px 6px 0; }
  .markdown-body blockquote p { margin: 4px 0; }
  .markdown-body table { width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 0.88em; background: var(--surface); border-radius: 8px; overflow: hidden; border: 1px solid var(--border); }
  .markdown-body th { background: #21262d; color: var(--text); padding: 10px 14px; text-align: left; font-weight: 600; white-space: nowrap; border-bottom: 2px solid var(--border); }
  .markdown-body td { padding: 8px 14px; border-bottom: 1px solid var(--border); color: var(--text); }
  .markdown-body tr:nth-child(even) td { background: var(--table-stripe); }
  .markdown-body tr:hover td { background: rgba(88,166,255,0.05); }
  .markdown-body code { background: var(--code-bg); padding: 2px 6px; border-radius: 4px; font-family: 'JetBrains Mono', 'Cascadia Code', 'Fira Code', monospace; font-size: 0.85em; color: var(--accent2); }
  .markdown-body pre { background: var(--code-bg); padding: 16px; border-radius: 8px; overflow-x: auto; margin: 12px 0; border: 1px solid var(--border); }
  .markdown-body pre code { background: none; padding: 0; color: var(--text); }
  .markdown-body pre code:not([class]) { color: var(--accent3); font-size: 0.82em; line-height: 1.4; }
  .markdown-body hr { border: none; border-top: 1px solid var(--border); margin: 32px 0; }
  .markdown-body strong { color: #fff; }
  .markdown-body em { color: var(--text-secondary); }
  .markdown-body ul, .markdown-body ol { padding-left: 24px; margin: 8px 0; }
  .markdown-body li { margin: 4px 0; }
  @media (max-width: 768px) { nav { width: 220px; min-width: 220px; } main { padding: 24px; } }
  @media (max-width: 600px) { body { flex-direction: column; } nav { width: 100%; min-width: auto; height: auto; position: static; border-right: none; border-bottom: 1px solid var(--border); padding: 12px; } nav .logo { padding-bottom: 12px; margin-bottom: 8px; } main { padding: 16px; } }
</style>
</head>
<body>
<nav>
  <div class="logo">🎵 音乐分析<span>报告</span></div>
  <div class="section-title">概览</div>
  <a href="#" data-page="index" class="active">📊 总览与对比</a>
  <div class="section-title">流派报告</div>
  <a href="#" data-page="genres/blues">🎸 Blues 蓝调</a>
  <a href="#" data-page="genres/classical">🎻 Classical 古典</a>
  <a href="#" data-page="genres/country">🤠 Country 乡村</a>
  <a href="#" data-page="genres/disco">🕺 Disco 迪斯科</a>
  <a href="#" data-page="genres/hiphop">🎤 Hip-Hop 嘻哈</a>
  <a href="#" data-page="genres/jazz">🎷 Jazz 爵士</a>
  <a href="#" data-page="genres/metal">🤘 Metal 金属</a>
  <a href="#" data-page="genres/pop">🎤 Pop 流行</a>
  <a href="#" data-page="genres/reggae">🌴 Reggae 雷鬼</a>
  <a href="#" data-page="genres/rock">🎸 Rock 摇滚</a>
</nav>
<main>
  <div id="content" class="markdown-body"></div>
</main>
<script>
// ── Embedded page data ──
__PAGES_PLACEHOLDER__

// ── Router ──
var contentEl = document.getElementById('content');
var navLinks = document.querySelectorAll('nav a[data-page]');

marked.setOptions({ breaks: true, gfm: true, headerIds: true, mangle: false });

function loadPage(page) {
  var md = PAGES[page] || PAGES['index'];
  contentEl.innerHTML = marked.parse(md);
  window.scrollTo(0, 0);
  navLinks.forEach(function(a) { a.classList.remove('active'); });
  var active = document.querySelector('nav a[data-page="' + page + '"]');
  if (active) active.classList.add('active');
  window.location.hash = page;
}

navLinks.forEach(function(link) {
  link.addEventListener('click', function(e) {
    e.preventDefault();
    loadPage(link.dataset.page);
  });
});

var hash = window.location.hash.slice(1);
loadPage(hash || 'index');

window.addEventListener('hashchange', function() {
  loadPage(window.location.hash.slice(1) || 'index');
});
</script>
</body>
</html>'''

# ── Combine ──
final_html = html.replace('__PAGES_PLACEHOLDER__', pages_js)

out_path = os.path.join(DOCS_DIR, 'viewer_standalone.html')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(final_html)

import sys
sys.stdout.reconfigure(encoding='utf-8')

size_kb = os.path.getsize(out_path) / 1024
print(f'[OK] Generated: viewer_standalone.html')
print(f'Size: {size_kb:.0f} KB')
print(f'Path: {out_path}')
print(f'Double-click to open in browser - no server needed!')
print(f'Embedded pages: {len(pages)} ({", ".join(pages.keys())})')
