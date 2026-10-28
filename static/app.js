let chartSelectionRate = null;
let chartTPR = null;
let chartTraining = null;

document.addEventListener('DOMContentLoaded', () => {
    // Bind buttons
    document.getElementById('btn-screen').addEventListener('click', screenResumes);
    document.getElementById('btn-run-experiment').addEventListener('click', runExperiment);
    document.getElementById('btn-run-audit').addEventListener('click', runAudit);
    document.getElementById('btn-download-csv').addEventListener('click', downloadCSV);
    document.getElementById('btn-train-pytorch').addEventListener('click', trainPyTorch);
    document.getElementById('btn-run-full-demo').addEventListener('click', runFullDemo);
    document.getElementById('btn-reset').addEventListener('click', resetDemo);

    // Initial load
    runAudit();
});

async function screenResumes() {
    const jobDesc = document.getElementById('job-description').value;
    const isDebiased = document.getElementById('toggle-debias').checked;

    const samplesResponse = await fetch('/api/sample_resumes');
    const sampleResumes = await samplesResponse.json();

    const response = await fetch('/api/screen', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            job_description: jobDesc,
            resumes: sampleResumes,
            is_debiased: isDebiased
        })
    });

    const data = await response.json();
    const tbody = document.getElementById('screening-table-body');
    tbody.innerHTML = '';

    data.results.forEach((item, index) => {
        const row = `<tr>
            <td><strong>#${index + 1}</strong></td>
            <td>${item.name}</td>
            <td><strong>${item.score}%</strong></td>
            <td><span class="${item.decision === 'Shortlisted' ? 'badge-success' : 'badge-warning'}">${item.decision}</span></td>
            <td>${item.matched_skills.join(', ') || 'None'}</td>
            <td>${item.missing_skills.join(', ') || 'None'}</td>
        </tr>`;
        tbody.innerHTML += row;
    });
}

async function runExperiment() {
    const jobDesc = document.getElementById('job-description').value;
    const maleText = document.getElementById('exp-male').value;
    const femaleText = document.getElementById('exp-female').value;

    const response = await fetch('/api/experiment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            job_description: jobDesc,
            resume_male: maleText,
            resume_female: femaleText
        })
    });

    const data = await response.json();
    
    const badge = document.getElementById('qualification-badge');
    if (data.qualifications_identical) {
        badge.className = 'badge-box badge-success';
        badge.innerText = `✅ Pre-Validation Passed: Resumes have identical qualifications (${data.text_similarity_between_resumes}% similarity).`;
    } else {
        badge.className = 'badge-box badge-warning';
        badge.innerText = `⚠️ Notice: Qualifications differ across resumes (${data.text_similarity_between_resumes}% similarity).`;
    }

    document.getElementById('baseline-results').innerHTML = `
        <p><strong>Male Candidate Score:</strong> ${data.baseline.male.score}% (${data.baseline.male.decision})</p>
        <p><strong>Female Candidate Score:</strong> ${data.baseline.female.score}% (${data.baseline.female.decision})</p>
        <p style="color:#e11d48"><strong>Bias Discrepancy Gap:</strong> ${data.baseline.score_gap} points</p>
    `;

    document.getElementById('debiased-results').innerHTML = `
        <p><strong>Male Candidate Score:</strong> ${data.debiased.male.score}% (${data.debiased.male.decision})</p>
        <p><strong>Female Candidate Score:</strong> ${data.debiased.female.score}% (${data.debiased.female.decision})</p>
        <p style="color:#16a34a"><strong>Bias Discrepancy Gap:</strong> ${data.debiased.score_gap} points (Fair)</p>
    `;
}

async function runAudit() {
    const response = await fetch('/api/audit');
    const audit = await response.json();

    const bm = audit.baseline_metrics;
    const dm = audit.debiased_metrics;

    document.getElementById('metrics-summary').innerHTML = `
        <div class="summary-card">
            <h4>Demographic Parity Gap</h4>
            <p>${bm.demographic_parity_difference} → <strong>${dm.demographic_parity_difference}</strong></p>
        </div>
        <div class="summary-card">
            <h4>Equal Opportunity Gap</h4>
            <p>${bm.equal_opportunity_difference} → <strong>${dm.equal_opportunity_difference}</strong></p>
        </div>
        <div class="summary-card">
            <h4>Task Accuracy</h4>
            <p>${(dm.accuracy * 100).toFixed(1)}%</p>
        </div>
        <div class="summary-card">
            <h4>Audited Candidates</h4>
            <p>${audit.total_candidates}</p>
        </div>
    `;

    renderAuditCharts(bm, dm);
}

function renderAuditCharts(bm, dm) {
    if (chartSelectionRate) chartSelectionRate.destroy();
    if (chartTPR) chartTPR.destroy();

    const ctxSR = document.getElementById('chart-selection-rate').getContext('2d');
    chartSelectionRate = new Chart(ctxSR, {
        type: 'bar',
        data: {
            labels: ['Male Selection Rate', 'Female Selection Rate'],
            datasets: [
                { label: 'Baseline (Biased)', data: [bm.selection_rate_male, bm.selection_rate_female], backgroundColor: '#f43f5e' },
                { label: 'Debiased (HireFair)', data: [dm.selection_rate_male, dm.selection_rate_female], backgroundColor: '#10b981' }
            ]
        },
        options: { responsive: true, plugins: { title: { display: true, text: 'Demographic Parity Check' } } }
    });

    const ctxTPR = document.getElementById('chart-tpr').getContext('2d');
    chartTPR = new Chart(ctxTPR, {
        type: 'bar',
        data: {
            labels: ['Male TPR', 'Female TPR'],
            datasets: [
                { label: 'Baseline (Biased)', data: [bm.tpr_male, bm.tpr_female], backgroundColor: '#fb7185' },
                { label: 'Debiased (HireFair)', data: [dm.tpr_male, dm.tpr_female], backgroundColor: '#34d399' }
            ]
        },
        options: { responsive: true, plugins: { title: { display: true, text: 'Equal Opportunity Check (TPR)' } } }
    });
}

async function trainPyTorch() {
    const response = await fetch('/api/train_debiaser', { method: 'POST' });
    const data = await response.json();
    
    const history = data.training_history;
    const epochs = history.map(h => h.epoch);
    const taskLoss = history.map(h => h.task_loss);
    const advLoss = history.map(h => h.adversary_loss);

    if (chartTraining) chartTraining.destroy();
    const ctx = document.getElementById('chart-training-loss').getContext('2d');
    chartTraining = new Chart(ctx, {
        type: 'line',
        data: {
            labels: epochs,
            datasets: [
                { label: 'Predictor Task Loss', data: taskLoss, borderColor: '#2563eb', fill: false },
                { label: 'Adversary Gender Loss', data: advLoss, borderColor: '#dc2626', fill: false }
            ]
        },
        options: { responsive: true, maintainAspectRatio: false, plugins: { title: { display: true, text: 'PyTorch Adversarial Debiasing Loss Curves' } } }
    });
}

function downloadCSV() {
    window.location.href = '/api/audit/download';
}

async function runFullDemo() {
    await screenResumes();
    await runExperiment();
    await runAudit();
    await trainPyTorch();
}

function resetDemo() {
    document.getElementById('toggle-debias').checked = false;
    screenResumes();
    runExperiment();
}