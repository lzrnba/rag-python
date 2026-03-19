# -*- coding: utf-8 -*-
path = r'e:\Project Design\project\python-rag\rag-python\frontend\index.html'

part1 = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>RAG 技术文档助手</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Noto+Sans+SC:wght@300;400;500&display=swap');
    :root{--bg:#0d1117;--bg2:#161b22;--bg3:#21262d;--border:#30363d;--accent:#58a6ff;--accent2:#3fb950;--accent3:#f78166;--text:#e6edf3;--text2:#8b949e;--tag-bg:#1f6feb22}
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:'Noto Sans SC',sans-serif;background:var(--bg);color:var(--text);height:100vh;display:flex;flex-direction:column;overflow:hidden}
    .navbar{display:flex;align-items:center;justify-content:space-between;padding:0 24px;height:56px;background:var(--bg2);border-bottom:1px solid var(--border);flex-shrink:0}
    .navbar-left{display:flex;align-items:center;gap:12px}
    .logo{width:32px;height:32px;background:linear-gradient(135deg,var(--accent),var(--accent2));border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:16px;font-weight:700;color:#0d1117;font-family:'JetBrains Mono',monospace}
    .title{font-size:15px;font-weight:500}.subtitle{font-size:11px;color:var(--text2);font-family:'JetBrains Mono',monospace}
    .badge{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--text2);font-family:'JetBrains Mono',monospace;padding:4px 12px;background:var(--bg3);border:1px solid var(--border);border-radius:20px}
    .dot{width:7px;height:7px;border-radius:50%;background:var(--accent3);animation:pulse 2s infinite}.dot.on{background:var(--accent2)}
    @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
    .main{display:flex;flex:1;overflow:hidden}
    .sidebar{width:248px;flex-shrink:0;background:var(--bg2);border-right:1px solid var(--border);display:flex;flex-direction:column;padding:14px 12px;gap:14px;overflow-y:auto}
    .sec-title{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--text2);padding:0 4px;font-family:'JetBrains Mono',monospace;margin-bottom:8px}
    .new-btn{width:100%;padding:10px 14px;background:var(--accent);color:#0d1117;border:none;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;display:flex;align-items:center;gap:8px;transition:opacity .15s;font-family:'Noto Sans SC',sans-serif}
    .new-btn:hover{opacity:.85}.new-btn:disabled{opacity:.5;cursor:not-allowed}
    .opt-row{display:flex;align-items:center;justify-content:space-between;font-size:12px;color:var(--text);padding:2px 4px}
    .opt-row input[type=range]{width:76px;accent-color:var(--accent)}
    .opt-val{font-family:'JetBrains Mono',monospace;color:var(--accent);font-size:12px;min-width:18px;text-align:right}
    .tog-row{display:flex;align-items:center;justify-content:space-between;font-size:12px;color:var(--text);padding:2px 4px}
    .tog{position:relative;width:34px;height:19px;cursor:pointer}.tog input{opacity:0;width:0;height:0}
    .tog-sl{position:absolute;inset:0;background:var(--border);border-radius:20px;transition:background .2s}
    .tog-sl::before{content:'';position:absolute;width:13px;height:13px;left:3px;top:3px;background:#fff;border-radius:50%;transition:transform .2s}
    .tog input:checked+.tog-sl{background:var(--accent2)}.tog input:checked+.tog-sl::before{transform:translateX(15px)}
    .hist-list{display:flex;flex-direction:column;gap:3px}
    .hist-item{padding:7px 10px;border-radius:6px;font-size:12px;color:var(--text2);cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;transition:background .15s}
    .hist-item:hover{background:var(--bg3);color:var(--text)}.hist-item.active{background:var(--tag-bg);color:var(--accent)}
    .dir-info{font-size:11px;color:var(--text2);font-family:'JetBrains Mono',monospace;padding:8px 10px;background:var(--bg3);border-radius:6px;border:1px solid var(--border);line-height:1.8}
    .dir-info b{color:var(--accent2)}
    .chat-area{flex:1;display:flex;flex-direction:column;overflow:hidden}
    .msgs{flex:1;overflow-y:auto;padding:20px 0;scroll-behavior:smooth}
    .msgs::-webkit-scrollbar{width:5px}.msgs::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
    .msg-row{display:flex;padding:5px 24px;gap:13px;max-width:840px;margin:0 auto;width:100%;animation:fi .2s ease}
    @keyframes fi{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
    .av{width:30px;height:30px;border-radius:8px;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;font-family:'JetBrains Mono',monospace}
    .av.user{background:linear-gradient(135deg,#1f4068,#2d6a9f);color:var(--accent)}.av.bot{background:linear-gradient(135deg,#1a2f1e,#2d5a36);color:var(--accent2)}
    .bbl{flex:1;min-width:0}.bbl-name{font-size:11px;color:var(--text2);margin-bottom:5px;font-family:'JetBrains Mono',monospace}
    .bbl-txt{font-size:14px;line-height:1.8;color:var(--text);white-space:pre-wrap;word-break:break-word}
    .bbl-txt code{font-family:'JetBrains Mono',monospace;background:var(--bg3);padding:2px 6px;border-radius:4px;font-size:13px;color:var(--accent)}
    .srcs{margin-top:10px;display:flex;flex-wrap:wrap;gap:6px}
    .src-tag{font-size:11px;padding:3px 10px;background:var(--tag-bg);border:1px solid #1f6feb55;border-radius:20px;color:var(--accent);font-family:'JetBrains Mono',monospace}
    .rsn-wrap{margin-top:8px}.rsn-wrap summary{font-size:11px;color:var(--accent2);font-family:'JetBrains Mono',monospace;cursor:pointer;list-style:none;padding:3px 0}
    .rsn-wrap summary::-webkit-details-marker{display:none}
    .rsn-wrap summary::before{content:"\\25B6  "}details[open] .rsn-wrap summary::before{content:"\\25BC  "}
    .rsn-body{margin-top:5px;padding:9px 13px;background:var(--bg3);border-left:3px solid var(--accent2);border-radius:0 6px 6px 0;font-size:12px;color:var(--text2);font-family:'JetBrains Mono',monospace;line-height:1.6;white-space:pre-wrap}
    .meta-row{display:flex;gap:14px;margin-top:8px;flex-wrap:wrap}.meta-c{font-size:11px;color:var(--text2);font-family:'JetBrains Mono',monospace}.meta-c b{color:var(--accent)}
    .typing{display:flex;gap:5px;align-items:center;padding:4px 0}
    .td{width:7px;height:7px;background:var(--text2);border-radius:50%;animation:bo 1.2s infinite}
    .td:nth-child(2){animation-delay:.2s}.td:nth-child(3){animation-delay:.4s}
    @keyframes bo{0%,80%,100%{transform:translateY(0);opacity:.4}40%{transform:translateY(-5px);opacity:1}}
    .welcome{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;gap:16px;color:var(--text2)}
    .wl-logo{width:60px;height:60px;background:linear-gradient(135deg,var(--accent),var(--accent2));border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:26px;font-weight:700;color:#0d1117;font-family:'JetBrains Mono',monospace}
    .welcome h2{font-size:19px;font-weight:500;color:var(--text)}.welcome p{font-size:13px;text-align:center;max-width:400px;line-height:1.7}
    .suggs{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;max-width:540px}
    .sugg{padding:7px 15px;background:var(--bg3);border:1px solid var(--border);border-radius:20px;font-size:12px;color:var(--text);cursor:pointer;transition:border-color .15s,color .15s}
    .sugg:hover{border-color:var(--accent);color:var(--accent)}
    .input-area{padding:13px 24px 16px;border-top:1px solid var(--border);background:var(--bg);flex-shrink:0}
    .input-wrap{max-width:840px;margin:0 auto;position:relative}
    .ibox{width:100%;background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:12px 48px 12px 15px;color:var(--text);font-size:14px;font-family:'Noto Sans SC',sans-serif;resize:none;outline:none;line-height:1.6;transition:border-color .15s;min-height:50px;max-height:140px}
    .ibox:focus{border-color:var(--accent)}.ibox::placeholder{color:var(--text2)}
    .send-btn{position:absolute;right:9px;bottom:9px;width:33px;height:33px;background:var(--accent);border:none;border-radius:8px;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:opacity .15s,transform .1s}
    .send-btn:hover{opacity:.85}.send-btn:active{transform:scale(.95)}.send-btn:disabled{opacity:.3;cursor:not-allowed}
    .hint{font-size:11px;color:var(--text2);margin-top:6px;text-align:center;font-family:'JetBrains Mono',monospace}
    .err-bar{margin:0 auto 8px;max-width:840px;padding:8px 13px;background:#3d1a1a;border:1px solid var(--accent3);border-radius:8px;font-size:12px;color:var(--accent3);font-family:'JetBrains Mono',monospace;display:none}
    .err-bar.show{display:block}
    @media(max-width:640px){.sidebar{display:none}}
  </style>
</head>
<body>'''

with open(path, 'w', encoding='utf-8') as f:
    f.write(part1)
print('Part 1 written')
