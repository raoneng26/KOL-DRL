/**
 * SimuMarket | 舆情-市场耦合仿真系统 核心逻辑
 * 模块：控制、拓扑、内容渲染、金融看板
 */

let network = null;
let nodes = new vis.DataSet([]);
let edges = new vis.DataSet([]);
let currentSentimentRecord = null; // 存储情绪分析结果供交易使用
let tradeTicker = null;            // 实时交易定时器

// 存储交易历史数据用于绘图
let chartData = {
    times: [],
    prices: [],
    pred_prices: [], // 增加预测线数据
    rl_worths: [],
    bnh_worths: []
};

const ROLE_THEME = {
    "意见领袖": { bg: '#fbbf24', border: '#d97706', label: 'KOL' },
    "专业分析师": { bg: '#3b82f6', border: '#1e40af', label: '分析师' },
    "积极交易者": { bg: '#10b981', border: '#064e3b', label: '交易员' },
    "恐慌型散户": { bg: '#f43f5e', border: '#881337', label: '恐慌者' },
    "其他观察者": { bg: '#94a3b8', border: '#475569', label: '观察员' }
};


// --- 页面加载完成后执行 ---
document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM fully loaded and parsed");
    
    // 1. 初始化金融图表
    if (typeof initFinanceCharts === 'function') initFinanceCharts();
    
    // 2. 初始化拖拽
    initDragLogic();

    // 3. 【修正重点】绑定文件上传显示
    const fileInput = document.getElementById('ui-config-file');
    const fileDisplay = document.getElementById('file-name-display');

    if (fileInput && fileDisplay) {
        fileInput.addEventListener('change', function(e) {
            if (this.files && this.files.length > 0) {
                const fileName = this.files[0].name;
                console.log("File selected:", fileName);
                // 更新界面显示文件名
                fileDisplay.innerHTML = `
                    <span class="text-[9px] font-bold text-blue-600 uppercase">
                        Selected: ${fileName}
                    </span>`;
                fileDisplay.classList.add('border-blue-400', 'bg-blue-50');
            }
        });
    } else {
        console.error("找不到文件上传组件 ID: ui-config-file 或 file-name-display");
    }
});

// --- 核心执行函数 ---
async function executeSim() {
    console.log("ExecuteSim triggered"); // 调试日志

    const contentArea = document.getElementById('ui-kol-content');
    const fileInput = document.getElementById('ui-config-file');
    const btn = document.getElementById('run-btn');
    const statusBar = document.getElementById('status-bar');
    const statusText = document.getElementById('status-text');

    // 1. 数据校验
    const content = contentArea ? contentArea.value : "";
    const file = (fileInput && fileInput.files) ? fileInput.files[0] : null;

    if (!content || !file) {
        console.warn("Input missing:", { content: !!content, file: !!file });
        // 视觉反馈：变红闪烁
        btn.classList.add('bg-rose-600');
        btn.innerText = "内容或文件缺失";
        setTimeout(() => {
            btn.classList.remove('bg-rose-600');
            btn.innerText = "Start Evolution";
        }, 2000);
        return;
    }

    // 2. 构造上传数据
    const formData = new FormData();
    formData.append('content', content);
    formData.append('config_file', file);

    // 3. UI 状态切换
    btn.disabled = true;
    btn.innerHTML = `<span class="animate-spin inline-block w-3 h-3 border-2 border-white border-t-transparent rounded-full mr-2"></span> Evolving...`;
    if(statusBar) statusBar.style.opacity = "0";

    try {
        console.log("Sending request to /api/run_simulation...");
        const response = await fetch('/api/run_simulation', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        
        const data = await response.json();
        console.log("Task accepted, ID:", data.task_id);

        // 4. 开始轮询状态
        pollStatus(data.task_id);

    } catch (error) {
        console.error("Simulation Start Failed:", error);
        btn.disabled = false;
        btn.innerText = "连接服务器失败";
        btn.classList.add('bg-rose-600');
    }
}

// --- 轮询状态函数 ---
function pollStatus(taskId) {
    const btn = document.getElementById('run-btn');
    const statusBar = document.getElementById('status-bar');
    const statusText = document.getElementById('status-text');

    const timer = setInterval(async () => {
        try {
            const res = await fetch(`/api/check_status/${taskId}`);
            const result = await res.json();

            console.log("Polling status:", result.status);

            if (result.status === "completed") {
                clearInterval(timer);
                btn.disabled = false;
                btn.innerHTML = "Start Evolution";
                if(statusText) statusText.innerText = "Evolution Completed";
                if(statusBar) statusBar.style.opacity = "1";
                
                // 自动刷新数据
                reloadDashboard();
                onSimulationComplete(); // 激活第二个按钮

            } else if (result.status === "failed") {
                clearInterval(timer);
                btn.disabled = false;
                btn.innerText = "仿真失败";
                console.error("Backend Error Log:", result.log);
            }
        } catch (e) {
            console.error("Polling error:", e);
        }
    }, 2000);
}

// --- 刷新看板数据 ---
async function reloadDashboard() {
    console.log("Refreshing dashboard...");
    try {
        // 获取日志
        const logRes = await fetch('/api/get_latest_log');
        const simData = await logRes.json();
        if (!simData.error) {
            renderSimulation(simData);
        }

        // 获取金融数据
        const marketRes = await fetch('/api/market_data');
        const marketData = await marketRes.json();
        renderFinanceCharts(marketData);
    } catch (e) {
        console.error("Reload failed:", e);
    }
}


// --- 4. 金融图表逻辑 (Plotly) ---
function initFinanceCharts() {
    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        showlegend: false,
        margin: { l: 40, r: 10, t: 10, b: 40 },
        xaxis: { gridcolor: '#f1f5f9', tickfont: {size: 9} },
        yaxis: { gridcolor: '#f1f5f9', tickfont: {size: 9} },
        autosize: true,
    };

    // 统一使用 price-chart 和 equity-chart
    // 价格图：预设两条线
    Plotly.newPlot('price-chart', [
        { x: [], y: [], name: 'Live Price', line: {color: '#94a3b8', width: 2} },
        { x: [], y: [], name: 'Prediction', line: {color: '#3b82f6', width: 2, dash: 'dot'} }
    ], layout);

    // 资产图：预设两条线
    Plotly.newPlot('equity-chart', [
        { x: [], y: [], name: 'RL Agent', fill: 'tozeroy', line: {color: '#f59e0b'} },
        { x: [], y: [], name: 'Buy & Hold', line: {color: '#cbd5e1', dash: 'dash'} }
    ], layout);
}

function renderFinanceCharts(data) {
    // 价格与预测
    const priceTrace = {
        x: data.times, y: data.real_price, name: 'Market',
        type: 'scatter', mode: 'lines', line: {color: '#cbd5e1', width: 2}
    };
    const predTrace = {
        x: data.times, y: data.pred_price, name: 'Pred',
        type: 'scatter', mode: 'lines', line: {color: '#3b82f6', width: 3, dash: 'dot'}
    };
    Plotly.react('finance-chart', [priceTrace, predTrace]);

    // 财富曲线对比
    const rlTrace = {
        x: data.times, y: data.rl_wealth, name: 'RL Agent',
        fill: 'tozeroy', type: 'scatter', line: {color: '#f97316', width: 3}
    };
    const holdTrace = {
        x: data.times, y: data.hold_wealth, name: 'Hold',
        type: 'scatter', line: {color: '#94a3b8', width: 1.5, dash: 'dash'}
    };
    Plotly.react('wealth-chart', [rlTrace, holdTrace]);
}

// --- 5. 辅助工具函数 ---
function parseRawPost(str) {
    try {
        const parsed = JSON.parse(str);
        return {
            content: parsed.content || str,
            user_name: parsed.user_name || "KOL_Source"
        };
    } catch (e) {
        // 处理非标准 JSON 或纯文本
        return { content: str, user_name: "KOL_Source" };
    }
}

/**
 * 根据文本内容对智能体分类并匹配颜色
 */
function getTheme(text) {
    if (!text) return ROLE_THEME.other;
    
    const t = text.toLowerCase();
    
    // 如果文本包含分析关键词
    if (t.includes('analysis') || t.includes('indicator') || t.includes('chart')) {
        return ROLE_THEME.analyst;
    }
    // 如果包含看涨/购买等积极关键词
    if (t.includes('buy') || t.includes('moon') || t.includes('bull') || t.includes('long')) {
        return ROLE_THEME.retail;
    }
    // 如果包含恐慌/卖出/暴跌等消极关键词
    if (t.includes('sell') || t.includes('drop') || t.includes('scam') || t.includes('short') || t.includes('crash')) {
        return ROLE_THEME.panic;
    }
    
    // 默认返回其他
    return ROLE_THEME.other;
}

function initNetworkGraph() {
    const container = document.getElementById('network-container');
    const options = {
        physics: { forceAtlas2Based: { gravitationalConstant: -50, springLength: 100 }, solver: 'forceAtlas2Based' },
        interaction: { hover: true }
    };
    if (!network) {
        network = new vis.Network(container, { nodes, edges }, options);
    }
}

// --- 6. 拖拽 UI 逻辑 ---
function initDragLogic() {
    const dragBox = document.getElementById("drag-console");
    const dragHeader = document.getElementById("drag-header");
    let isDragging = false;
    let offset = [0, 0];

    dragHeader.onmousedown = (e) => {
        isDragging = true;
        offset = [dragBox.offsetLeft - e.clientX, dragBox.offsetTop - e.clientY];
    };

    document.onmousemove = (e) => {
        if (!isDragging) return;
        dragBox.style.left = (e.clientX + offset[0]) + "px";
        dragBox.style.top = (e.clientY + offset[1]) + "px";
    };

    document.onmouseup = () => isDragging = false;
}

function toggleConsole() {
    const body = document.getElementById('console-body');
    body.classList.toggle('hidden');
}


/**
 * 全量数据渲染
 */
function renderSimulation(data) {
    console.log("DEBUG: 开始渲染数据", data);

    const kolContentEl = document.getElementById('kol-content');
    const flowContainer = document.getElementById('comment-flow');
    
    if (!kolContentEl || !flowContainer) {
        console.error("DOM 容器未找到");
        return;
    }

    // 1. 渲染 KOL 内容
    const postParsed = parseRawPost(data.post_content || data.content);
    kolContentEl.innerText = postParsed.content;
    document.getElementById('kol-name').innerText = postParsed.user_name || "KOL_Source";
    document.getElementById('stat-likes').innerText = data.post_likes || 0;
    document.getElementById('stat-rt').innerText = data.post_retweets || 0;

    // 2. 清空容器并准备数据集
    flowContainer.innerHTML = ''; 
    nodes.clear();
    edges.clear();

    // 添加核心 KOL 节点
    nodes.add({
        id: 'kol', label: 'KOL', shape: 'star', size: 25,
        color: { background: '#fbbf24', border: '#d97706' },
        font: { size: 14, weight: '900' }
    });

    const comments = data.comments || [];
    document.getElementById('stat-cm').innerText = comments.length;

    // 3. 循环渲染评论 (使用 try-catch 防止单个错误导致整个循环崩溃)
    comments.forEach((c, idx) => {
        try {
            const agentId = String(c.agent_id || `a_${idx}`);
            const theme = getTheme(c.text || "");
            const subComments = c.sub_comments || [];

            // A. 构建卡片 HTML
            const card = document.createElement('div');
            card.className = "bg-white p-6 rounded-[2.5rem] shadow-sm border border-slate-50 mb-4";
            
            let subHtml = "";
            if (subComments.length > 0) {
                subHtml = `<div class="guide-line mt-4 space-y-3">`;
                subComments.forEach((sub, sIdx) => {
                    const subId = String(sub.agent_id || `${agentId}_s${sIdx}`);
                    const eId = `e_${subId}_${agentId}`;
                    subHtml += `
                        <div class="sub-comment-item py-2 px-2 rounded-lg" 
                             onmouseenter="focusInteraction('${subId}', '${agentId}', '${eId}')" 
                             onmouseleave="resetInteraction()">
                            <div class="text-[8px] font-bold text-slate-400">Agent #${subId}</div>
                            <p class="text-slate-500 text-xs">${sub.text}</p>
                        </div>`;
                    
                    // 二级节点入网
                    nodes.add({ id: subId, shape: 'dot', size: 8, color: '#cbd5e1', initialColor: '#cbd5e1' });
                    edges.add({ id: eId, from: subId, to: agentId, dashes: true, color: '#cbd5e1' });
                });
                subHtml += `</div>`;
            }

            card.innerHTML = `
                <div class="flex items-center gap-2 mb-2">
                    <span class="w-2 h-2 rounded-full" style="background:${theme.bg}"></span>
                    <span class="font-bold text-[9px] text-slate-400 uppercase">Agent #${agentId}</span>
                </div>
                <p class="text-slate-700 text-sm font-semibold">${c.text}</p>
                <div class="mt-4 flex gap-4 text-[9px] font-black text-slate-200 border-t border-slate-50 pt-2">
                    <span>❤ ${c.likes}</span> <span>🔄 ${c.retweets}</span> <span class="text-blue-400">💬 ${subComments.length}</span>
                </div>
                ${subHtml}`;
            
            flowContainer.appendChild(card);

            // B. 一级节点入网
            nodes.add({
                id: agentId, label: agentId, shape: 'dot', size: 15,
                color: { background: theme.bg, border: theme.border },
                initialColor: theme.bg
            });
            edges.add({ id: `e_${agentId}_kol`, from: agentId, to: 'kol', color: '#e2e8f0' });

        } catch (err) {
            console.error("渲染单条评论出错:", err);
        }
    });

    // 4. 强制延迟加载网络图 (给 DOM 留出渲染时间)
    setTimeout(() => {
        initNetworkGraph();
    }, 300);
}

function initNetworkGraph() {
    const container = document.getElementById('network-container');
    if (!container) return;

    console.log("DEBUG: 正在初始化网络图，容器高度:", container.offsetHeight);

    const options = {
        physics: {
            enabled: true,
            solver: 'forceAtlas2Based',
            forceAtlas2Based: { gravitationalConstant: -100, springLength: 150 }
        },
        interaction: { hover: true, zoomView: true, dragView: true }
    };

    if (network) {
        network.destroy(); // 彻底销毁旧实例防止内存溢出
    }
    network = new vis.Network(container, { nodes, edges }, options);
    
    // 强制缩放
    network.once('stabilized', () => {
        network.fit();
    });
}

// --- 核心渲染函数 ---
function renderSimulation(data) {
    console.log("开始深度渲染...");
    
    // 1. 基础 UI 更新
    const postParsed = parseRawPost(data.post_content || data.content);
    document.getElementById('kol-content').innerText = postParsed.content;
    document.getElementById('kol-name').innerText = postParsed.user_name || "KOL_Source";
    document.getElementById('stat-likes').innerText = data.post_likes || 0;
    document.getElementById('stat-rt').innerText = data.post_retweets || 0;
    
    const comments = data.comments || [];
    document.getElementById('stat-cm').innerText = comments.length;

    // 2. 清空并准备数据集
    const flowContainer = document.getElementById('comment-flow');
    flowContainer.innerHTML = '';
    nodes.clear();
    edges.clear();

    // 注入 KOL 核心节点
    nodes.add({
        id: 'kol', label: postParsed.user_name || 'KOL', shape: 'star', size: 30,
        color: { background: '#fbbf24', border: '#d97706' },
        font: { size: 14, weight: 'bold', color: '#92400e' },
        fixed: true
    });

    // 3. 遍历渲染 (全量)
    comments.forEach((c, idx) => {
        const agentId = String(c.agent_id || `a_${idx}`);
        const theme = getTheme(c.text);
        const subComments = c.sub_comments || [];
        const repliesId = `replies-${idx}`;

        // A. 网络图节点 (带分类和名称)
        if (!nodes.get(agentId)) {
            nodes.add({
                id: agentId, 
                label: `${theme.label}\n#${agentId}`, // 显示分类和ID
                shape: 'dot', size: 20,
                color: { background: theme.bg, border: theme.border },
                font: { size: 11, color: '#334155' },
                initialColor: theme.bg
            });
            edges.add({ id: `e_${agentId}`, from: agentId, to: 'kol', color: '#e2e8f0', width: 2 });
        }

        // B. 构建社交卡片 (带折叠逻辑)
        const card = document.createElement('div');
        card.className = "bg-white p-6 rounded-[2.5rem] shadow-sm border border-slate-50 mb-4 transition-all";
        
        let subHtml = "";
        if (subComments.length > 0) {
            subHtml = `
                <div class="flex items-center justify-between mt-4 pt-4 border-t border-slate-50">
                    <span class="text-[9px] font-black text-blue-500 uppercase tracking-widest">${subComments.length} 互动反馈</span>
                    <button onclick="toggleReplies('${repliesId}', this)" class="text-[9px] font-black text-slate-400 hover:text-blue-600 uppercase transition-all">展开互动 +</button>
                </div>
                <div id="${repliesId}" class="replies-container">
                    <div class="guide-line mt-4 space-y-4">`;
            
            subComments.forEach((sub, sIdx) => {
                const subId = String(sub.agent_id || `${agentId}_s${sIdx}`);
                const edgeId = `e_${subId}_${agentId}`;
                const stheme = getTheme(sub.text);

                subHtml += `
                    <div class="sub-comment-item py-2 px-1 rounded-lg" 
                         onmouseenter="focusInteraction('${subId}', '${agentId}', '${edgeId}')" 
                         onmouseleave="resetInteraction()">
                        <div class="font-black text-[8px] mb-1" style="color:${stheme.bg}">${stheme.label} #${subId}</div>
                        <p class="text-slate-500 text-xs font-medium">${sub.text}</p>
                    </div>`;
                
                // 二级节点入网
                if (!nodes.get(subId)) {
                    nodes.add({
                        id: subId, label: subId, shape: 'dot', size: 10,
                        color: { background: '#cbd5e1', border: '#94a3b8' },
                        font: { size: 8, color: '#94a3b8' },
                        initialColor: '#cbd5e1'
                    });
                    edges.add({ id: edgeId, from: subId, to: agentId, dashes: true, color: '#cbd5e1' });
                }
            });
            subHtml += `</div></div>`;
        }

        card.innerHTML = `
            <div class="flex items-center gap-2 mb-3">
                <span class="w-2 h-2 rounded-full shadow-sm" style="background:${theme.bg}"></span>
                <span class="font-black text-[9px] uppercase tracking-widest text-slate-400">${theme.label} #${agentId}</span>
            </div>
            <p class="text-slate-800 text-sm font-semibold leading-relaxed">${c.text}</p>
            ${subHtml}
        `;
        flowContainer.appendChild(card);
    });

    // 4. 强力刷新网络图
    setTimeout(() => {
        initNetworkGraph();
    }, 500);
}

// --- 交互逻辑 ---
/**
 * SimuMarket | 核心仿真渲染脚本 (深度复刻版)
 */




// 辅助函数：根据角色获取主题
function getThemeByRole(role) {
    return ROLE_THEME[role] || ROLE_THEME["其他观察者"];
}

// --- 2. 核心渲染函数 ---
function renderSimulation(data) {
    console.log("开始渲染仿真数据...");

    // A. 基础内容填充
    const postParsed = parseRawPost(data.post_content || data.content);
    document.getElementById('kol-content').innerText = postParsed.content;
    document.getElementById('kol-name').innerText = postParsed.user_name || "KOL";
    
    // B. 【修复指标显示】
    document.getElementById('stat-likes').innerText = data.post_likes || 0;
    document.getElementById('stat-rt').innerText = data.post_retweets || 0;
    const comments = data.comments || [];
    document.getElementById('stat-cm').innerText = comments.length; // 评论数

    // C. 清空容器与数据集
    const flowContainer = document.getElementById('comment-flow');
    flowContainer.innerHTML = '';
    nodes.clear();
    edges.clear();

    // D. 注入中心 KOL 节点
    nodes.add({
        id: 'kol', label: postParsed.user_name, shape: 'star', size: 30,
        color: { background: '#fbbf24', border: '#d97706' },
        font: { size: 14, weight: 'bold', color: '#92400e' },
        fixed: true
    });

    // E. 遍历渲染全量评论
    comments.forEach((c, idx) => {
        const agentId = String(c.agent_id);
        const role = c.agent_role || "其他观察者"; // 使用数据中的 agent_role
        const theme = getThemeByRole(role);
        const subComments = c.sub_comments || [];
        const repliesId = `replies-${idx}`;

        // 1. 网络图节点：显示角色名称和 ID
        if (!nodes.get(agentId)) {
            nodes.add({
                id: agentId, 
                label: `${theme.label}\n#${agentId}`, 
                shape: 'dot', size: 20,
                color: { background: theme.bg, border: theme.border },
                font: { size: 10, color: '#475569' },
                initialColor: theme.bg,
                initialBorder: theme.border
            });
            edges.add({ id: `e_main_${agentId}`, from: agentId, to: 'kol', color: '#e2e8f0', width: 2 });
        }

        // 2. 社交卡片 HTML (复刻原 HTML 的二级折叠和样式)
        const card = document.createElement('div');
        card.className = "bg-white p-8 rounded-[2.5rem] shadow-sm border border-slate-50 mb-6 transition-all";
        
        let subHtml = "";
        if (subComments.length > 0) {
            subHtml = `
                <div class="flex items-center justify-between mt-6 pt-4 border-t border-slate-50">
                    <span class="text-[9px] font-black text-blue-500 uppercase tracking-widest">${subComments.length} 条互动回复</span>
                    <button onclick="toggleReplies('${repliesId}', this)" class="text-[9px] font-black text-slate-400 hover:text-blue-600 uppercase transition-all">展开互动 +</button>
                </div>
                <div id="${repliesId}" class="replies-container">
                    <div class="guide-line mt-4 space-y-4">`;
            
            subComments.forEach((sub, sIdx) => {
                const subId = String(sub.agent_id);
                const subRole = sub.agent_role || "其他观察者";
                const subTheme = getThemeByRole(subRole);
                const edgeId = `e_sub_${subId}_${agentId}`;

                subHtml += `
                    <div class="sub-comment-item py-2 px-2 rounded-xl" 
                         onmouseenter="focusInteraction('${subId}', '${agentId}', '${edgeId}')" 
                         onmouseleave="resetInteraction()">
                        <div class="flex items-center gap-2 mb-1">
                            <span class="w-1.5 h-1.5 rounded-full" style="background:${subTheme.bg}"></span>
                            <div class="font-black text-[8px] uppercase tracking-tighter" style="color:${subTheme.bg}">${subRole} #${subId}</div>
                        </div>
                        <p class="text-slate-500 text-xs font-medium leading-relaxed">${sub.text}</p>
                    </div>`;
                
                // 二级节点入网
                if (!nodes.get(subId)) {
                    nodes.add({
                        id: subId, label: subId, shape: 'dot', size: 10,
                        color: { background: '#f1f5f9', border: '#cbd5e1' },
                        font: { size: 8, color: '#94a3b8' },
                        initialColor: '#f1f5f9',
                        initialBorder: '#cbd5e1'
                    });
                    edges.add({ id: edgeId, from: subId, to: agentId, dashes: true, color: '#e2e8f0' });
                }
            });
            subHtml += `</div></div>`;
        }

        card.innerHTML = `
            <div class="flex items-center gap-2 mb-3">
                <span class="w-2.5 h-2.5 rounded-full shadow-sm" style="background:${theme.bg}"></span>
                <span class="font-black text-[10px] uppercase tracking-[0.1em] text-slate-800">${role} #${agentId}</span>
            </div>
            <p class="text-slate-800 text-sm font-semibold leading-relaxed mb-4">${c.text}</p>
            <div class="flex gap-6 text-[10px] font-black text-slate-300 uppercase italic">
                <span>❤ ${c.likes || 0}</span>
                <span>🔄 ${c.retweets || 0}</span>
            </div>
            ${subHtml}
        `;
        flowContainer.appendChild(card);
    });

    // F. 初始化网络图
    setTimeout(() => {
        initNetworkGraph();
    }, 300);
}

// --- 3. 交互逻辑复刻 ---

function toggleReplies(id, btn) {
    const el = document.getElementById(id);
    const expanded = el.classList.toggle('expanded');
    btn.innerText = expanded ? "收起互动 -" : "展开互动 +";
    btn.classList.toggle('bg-slate-50', expanded);
    btn.classList.toggle('px-3', expanded);
    btn.classList.toggle('py-1', expanded);
    btn.classList.toggle('rounded-lg', expanded);
}

function focusInteraction(subId, parentId, edgeId) {
    if(!network) return;
    // 高亮子节点
    nodes.update({ 
        id: subId, size: 25, 
        color: { background: '#f59e0b', border: '#d97706' },
        font: { size: 14, color: '#f59e0b', weight: 'bold' } 
    });
    // 对父节点增加光晕
    nodes.update({ id: parentId, shadow: { enabled: true, color: '#f59e0b', size: 25 } });
    // 变厚连线
    edges.update({ id: edgeId, color: '#f59e0b', width: 4, dashes: false });
    // 视角平滑移动
    network.fit({ nodes: [subId, parentId], animation: true });
}

function resetInteraction() {
    if(!network) return;
    nodes.forEach(n => {
        if(n.id === 'kol') return;
        const isSub = n.size === 25 || n.id.includes('_s'); 
        nodes.update({ 
            id: n.id, 
            size: n.initialColor === '#f1f5f9' ? 10 : 20, 
            color: { background: n.initialColor, border: n.initialBorder },
            font: { size: n.initialColor === '#f1f5f9' ? 8 : 10, color: '#475569', weight: 'normal' },
            shadow: { enabled: false } 
        });
    });
    edges.forEach(e => {
        edges.update({ id: e.id, color: '#e2e8f0', width: e.id.includes('main') ? 2 : 1, dashes: e.id.includes('sub') });
    });
}

function initNetworkGraph() {
    const container = document.getElementById('network-container');
    if (!container) return;

    const options = {
        physics: {
            enabled: true,
            solver: 'forceAtlas2Based',
            forceAtlas2Based: { gravitationalConstant: -120, springLength: 180, springConstant: 0.05 },
            stabilization: { iterations: 150 }
        },
        interaction: { hover: true, zoomView: true, dragView: true },
        nodes: { borderWidth: 2, font: { face: 'Inter' } },
        edges: { selectionWidth: 2 }
    };

    if (network) network.destroy();
    network = new vis.Network(container, { nodes, edges }, options);
    network.once('stabilized', () => network.fit());
}

function updateSentimentUI(res) {
    const data = res.sentiment_result;
    if (!data) return;

    // 1. 更新 Gemo 数值与颜色
    const gemo = data.gemo.toFixed(2);
    const gemoEl = document.getElementById('gemo-value');
    const gemoLabel = document.getElementById('gemo-label');
    
    gemoEl.innerText = gemo;
    if (gemo > 0.1) {
        gemoEl.className = "text-3xl font-black text-emerald-500";
        gemoLabel.innerText = "OPTIMISTIC";
    } else if (gemo < -0.1) {
        gemoEl.className = "text-3xl font-black text-rose-500";
        gemoLabel.innerText = "PESSIMISTIC";
    } else {
        gemoEl.className = "text-3xl font-black text-slate-900";
        gemoLabel.innerText = "NEUTRAL";
    }

    // 2. 更新 KOL FinBERT 进度条
    const pos = (data.sent_pos * 100).toFixed(1);
    const neg = (data.sent_neg * 100).toFixed(1);
    const neu = (data.sent_neu * 100).toFixed(1);

    document.getElementById('v-pos').innerText = pos + "%";
    document.getElementById('b-pos').style.width = pos + "%";
    document.getElementById('v-neg').innerText = neg + "%";
    document.getElementById('b-neg').style.width = neg + "%";
    
    console.log("Sentiment processing records:", data);
}


// 1. 在仿真完成后的回调中激活情绪按钮

// 在 update 进度条时增加最小像素宽度逻辑
const val_pos = d.sent_pos * 100;
const val_neg = d.sent_neg * 100;
const val_neu = d.sent_neu * 100;

// 更新百分比文字
document.getElementById('txt-pos').innerText = val_pos.toFixed(1) + "%";
document.getElementById('txt-neg').innerText = val_neg.toFixed(1) + "%";
document.getElementById('txt-neu').innerText = val_neu.toFixed(1) + "%";

// 更新进度条宽度：如果值大于0但很小，至少给 2% 的宽度让它可见
document.getElementById('bar-pos').style.width = (val_pos > 0 && val_pos < 2 ? 2 : val_pos) + "%";
document.getElementById('bar-neg').style.width = (val_neg > 0 && val_neg < 2 ? 2 : val_neg) + "%";
document.getElementById('bar-neu').style.width = (val_neu > 0 && val_neu < 2 ? 2 : val_neu) + "%";

function onSimulationComplete() {
    const sentBtn = document.getElementById('sent-btn');
    sentBtn.disabled = false;
    sentBtn.classList.remove('bg-slate-200', 'text-slate-400', 'cursor-not-allowed');
    sentBtn.classList.add('bg-blue-600', 'text-white', 'shadow-blue-200');
    sentBtn.innerHTML = "Analyze Sentiment";
}

// 2. 手动触发情绪处理的函数
async function analyzeSentiment() {
    const btn = document.getElementById('sent-btn');
    const area = document.getElementById('sentiment-results-area');
    
    btn.disabled = true;
    btn.innerHTML = `<span>Processing...</span>`;

    try {
        const response = await fetch('/api/process_sentiment', { method: 'POST' });
        const result = await response.json();

        if (result.status === "success") {
            const data = result.data; // 确保这里定义了 data
            currentSentimentRecord = data; // 存入全局变量

            if (area) area.style.opacity = "1";
            
            // 使用 data 而不是 d
            document.getElementById('gemo-val').innerText = data.gemo.toFixed(3);
            
            const p_pos = (data.sent_pos * 100).toFixed(1) + "%";
            const p_neg = (data.sent_neg * 100).toFixed(1) + "%";
            const p_neu = (data.sent_neu * 100).toFixed(1) + "%";

            document.getElementById('bar-pos').style.width = p_pos;
            document.getElementById('bar-neg').style.width = p_neg;
            document.getElementById('bar-neu').style.width = p_neu;
            
            document.getElementById('txt-pos').innerText = p_pos;
            document.getElementById('txt-neg').innerText = p_neg;
            document.getElementById('txt-neu').innerText = p_neu;

            btn.innerHTML = "Analysis Complete";
            btn.className = "w-full mt-3 py-3 bg-emerald-500 text-white rounded-xl font-black text-[9px] uppercase tracking-[0.3em]";
            
            // 激活交易按钮
            const tradeBtn = document.getElementById('trade-btn');
            if (tradeBtn) {
                tradeBtn.disabled = false;
                tradeBtn.className = "px-4 py-2 bg-slate-900 text-white text-[9px] font-black rounded-xl uppercase tracking-widest cursor-pointer";
            }
        }
    } catch (e) {
        console.error("Sentiment Error:", e);
        btn.disabled = false;
    }
}



// --- 2. 修正后的交易执行函数 ---
async function executeTrade() {
    console.log("尝试启动实时决策...");
    const btn = document.getElementById('trade-btn');
    
    if (!currentSentimentRecord) {
        alert("请先完成情绪分析！");
        return;
    }

    // 重置全局数据缓存
    chartData = { times: [], prices: [], pred_prices: [], rl_worths: [], bnh_worths: [] };

    // UI 状态
    btn.innerHTML = `<span class="animate-spin inline-block w-3 h-3 border-2 border-white border-t-transparent rounded-full mr-2"></span> Connecting Bybit...`;
    btn.disabled = true;

    try {
        // 通知后端启动 Session
        const startRes = await fetch('/api/start_trade_session', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ sentiment_record: currentSentimentRecord })
        });
        const startResult = await startRes.json();
        console.log("会话启动结果:", startResult);

        btn.innerHTML = "Live Mode ON";
        btn.className = "px-4 py-2 bg-emerald-500 text-white text-[9px] font-black rounded-xl uppercase tracking-widest";

        // 清除旧定时器
        if (tradeTicker) clearInterval(tradeTicker);

        // 启动定时器：每 5 秒获取一次数据
        tradeTicker = setInterval(async () => {
            try {
                const res = await fetch('/api/get_trade_tick');
                if (!res.ok) throw new Error("后端接口报错");
                const tick = await res.json();
                
                console.log("收到实时 Tick:", tick); // 调试日志：如果控制台没打印这个，说明没取到数

                if (tick.price) {
                    updateLiveCharts(tick);
                }
            } catch (err) {
                console.error("轮询 Tick 失败:", err);
            }
        }, 5000);

    } catch (e) {
        console.error("启动决策失败:", e);
        btn.innerHTML = "Retry Decision";
        btn.disabled = false;
    }
}

// --- 3. 核心：更新图表函数 (改用 React 方案) ---
function updateLiveCharts(tick) {
    // 追加数据到全局缓存
    chartData.times.push(tick.timestamp);
    chartData.prices.push(tick.price);
    chartData.pred_prices.push(tick.pred_price || tick.price * 1.001); // 容错处理
    chartData.rl_worths.push(tick.rl_worth);
    chartData.bnh_worths.push(tick.bnh_worth);

    const commonLayout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        showlegend: true,
        legend: { orientation: 'h', y: 1.1, font: {size: 8} },
        margin: { l: 40, r: 10, t: 10, b: 40 },
        xaxis: { gridcolor: '#f1f5f9', tickfont: {size: 8} },
        yaxis: { gridcolor: '#f1f5f9', tickfont: {size: 8} }
    };

    // 更新价格看板 (实盘 + 预测)
    Plotly.react('price-chart', [
        { 
            x: chartData.times, y: chartData.prices, 
            name: 'Bybit Price', line: {color: '#94a3b8', width: 2} 
        },
        { 
            x: chartData.times, y: chartData.pred_prices, 
            name: 'RL Prediction', line: {color: '#3b82f6', width: 2, dash: 'dot'} 
        }
    ], commonLayout);

    // 更新资产曲线看板 (RL vs Buy&Hold)
    Plotly.react('equity-chart', [
        { 
            x: chartData.times, y: chartData.rl_worths, 
            name: 'RL Strategy', fill: 'tozeroy', line: {color: '#f59e0b', width: 3} 
        },
        { 
            x: chartData.times, y: chartData.bnh_worths, 
            name: 'Market (B&H)', line: {color: '#cbd5e1', dash: 'dash', width: 1.5} 
        }
    ], commonLayout);

    // 更新 Alpha 收益文字
    const alpha = ((tick.rl_worth / tick.bnh_worth - 1) * 100).toFixed(2);
    const mAlpha = document.getElementById('m-alpha');
    if (mAlpha) {
        mAlpha.innerText = (alpha > 0 ? "+" : "") + alpha + "%";
        mAlpha.className = alpha >= 0 ? "text-xs font-black text-emerald-600" : "text-xs font-black text-rose-600";
    }
}
// --- 辅助函数：计算金融指标 ---
function calculateLiveMetrics() {
    if (chartData.rl_worths.length < 2) return;

    const currentRL = chartData.rl_worths[chartData.rl_worths.length - 1];
    const currentBNH = chartData.bnh_worths[chartData.bnh_worths.length - 1];

    // 1. ALPHA 计算
    const alpha = ((currentRL / currentBNH - 1) * 100).toFixed(2);
    document.getElementById('m-alpha').innerText = (alpha > 0 ? "+" : "") + alpha + "%";

    // 2. DRAWDOWN 计算
    const peak = Math.max(...chartData.rl_worths);
    const mdd = ((currentRL / peak - 1) * 100).toFixed(2);
    document.getElementById('m-mdd').innerText = mdd + "%";

    // 3. SHARPE 计算 (每分钟数据年化)
    // 计算收益率序列
    const returns = [];
    for (let i = 1; i < chartData.rl_worths.length; i++) {
        returns.push((chartData.rl_worths[i] / chartData.rl_worths[i-1]) - 1);
    }
    const avgRet = returns.reduce((a, b) => a + b) / returns.length;
    const stdDev = Math.sqrt(returns.map(x => Math.pow(x - avgRet, 2)).reduce((a, b) => a + b) / returns.length);
    
    if (stdDev !== 0) {
        // 年化系数: 分钟转年 ≈ sqrt(365*24*60)
        const sharpe = (avgRet / stdDev) * Math.sqrt(525600);
        document.getElementById('m-sharpe').innerText = sharpe.toFixed(2);
    }
}

// --- 核心更新逻辑 ---
function updateLiveCharts(tick) {
    chartData.times.push(tick.timestamp);
    chartData.prices.push(tick.price);
    chartData.pred_prices.push(tick.pred_price);
    chartData.rl_worths.push(tick.rl_worth);
    chartData.bnh_worths.push(tick.bnh_worth);

    // 1. 价格看板配置
    const priceLayout = {
        title: { text: 'Market Price (USDT)', font: { size: 10, color: '#94a3b8' } },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { l: 50, r: 10, t: 30, b: 40 },
        xaxis: { gridcolor: '#f1f5f9', tickfont: {size: 8} },
        // 【核心修复】不包含0点，自动聚焦波动范围
        yaxis: { 
            gridcolor: '#f1f5f9', 
            tickfont: {size: 8},
            autorange: true,
            fixedrange: false,
            title: 'Price / USDT'
        }
    };

    Plotly.react('price-chart', [
        { x: chartData.times, y: chartData.prices, name: 'Live Price', line: {color: '#94a3b8', width: 2} },
        { x: chartData.times, y: chartData.pred_prices, name: 'Prediction', line: {color: '#3b82f6', width: 2, dash: 'dot'} }
    ], priceLayout);

    // 2. 资产看板配置
    const equityLayout = {
        title: { text: 'Portfolio Net Worth (Index)', font: { size: 10, color: '#94a3b8' } },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { l: 50, r: 10, t: 30, b: 40 },
        xaxis: { gridcolor: '#f1f5f9', tickfont: {size: 8} },
        yaxis: { 
            gridcolor: '#f1f5f9', 
            tickfont: {size: 8},
            autorange: true, // 【核心修复】让 10.004 和 9.992 看起来波动明显
            title: 'Worth / USDT'
        }
    };

    Plotly.react('equity-chart', [
        { x: chartData.times, y: chartData.rl_worths, name: 'RL Agent', fill: 'tozeroy', line: {color: '#f59e0b', width: 3} },
        { x: chartData.times, y: chartData.bnh_worths, name: 'Benchmark', line: {color: '#cbd5e1', dash: 'dash', width: 1.5} }
    ], equityLayout);

    // 3. 执行指标计算
    calculateLiveMetrics();
}
// 辅助函数：在图表上打标记
function addTradeMarker(time, price, text, color, symbol) {
    const annotation = {
        x: time, y: price, text: text,
        showarrow: true, arrowhead: 2, arrowcolor: color,
        ax: 0, ay: symbol === 'triangle-up' ? 30 : -30,
        font: { color: color, size: 9, weight: 'bold' },
        bgcolor: 'white', bordercolor: color
    };
    Plotly.relayout('price-chart', {
        'annotations': (Plotly.graphData[0].layout.annotations || []).concat([annotation])
    });
}