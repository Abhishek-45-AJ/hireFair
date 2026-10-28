import json
import os
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from models.resume_scorer import score_single_resume, verify_identical_qualifications
from models.adversarial_debiaser import train_pytorch_adversarial_model
from fairness.audit import run_fairness_audit, generate_synthetic_candidate_dataset

app = Flask(__name__)

# Pre-train PyTorch model on application startup
print("[INIT] Pre-training PyTorch Adversarial Debiasing Model...")
train_pytorch_adversarial_model(epochs=30)
print("[INIT] Model pre-training complete.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/sample_resumes', methods=['GET'])
def get_sample_resumes():
    sample_file_path = 'data/sample_resumes.json'
    if not os.path.exists(sample_file_path):
        return jsonify({'error': 'Sample resumes dataset not found.'}), 440
    with open(sample_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return jsonify(data)

@app.route('/api/screen', methods=['POST'])
def screen_resumes():
    data = request.json or {}
    job_desc = data.get('job_description', '')
    resumes = data.get('resumes', [])
    is_debiased = data.get('is_debiased', False)

    if not job_desc or not resumes:
        return jsonify({'error': 'Job description and candidate resumes are required.'}), 400

    results = []
    for item in resumes:
        r_text = item.get('text', '')
        r_name = item.get('name', 'Candidate')
        
        if not r_text.strip():
            continue

        res = score_single_resume(r_text, job_desc, is_debiased=is_debiased)
        res['name'] = r_name
        results.append(res)

    results.sort(key=lambda x: x['score'], reverse=True)
    return jsonify({'results': results})

@app.route('/api/experiment', methods=['POST'])
def run_gender_experiment():
    data = request.json or {}
    job_desc = data.get('job_description', '')
    res_male = data.get('resume_male', '')
    res_female = data.get('resume_female', '')

    if not job_desc or not res_male or not res_female:
        return jsonify({'error': 'Job description and both male/female resume texts are required.'}), 400

    is_identical, sim, skills1, skills2 = verify_identical_qualifications(res_male, res_female)
    
    # Baseline Scoring (Unfair/Biased)
    base_male = score_single_resume(res_male, job_desc, is_debiased=False)
    base_female = score_single_resume(res_female, job_desc, is_debiased=False)
    
    # Debiased Scoring (HireFair)
    deb_male = score_single_resume(res_male, job_desc, is_debiased=True)
    deb_female = score_single_resume(res_female, job_desc, is_debiased=True)

    return jsonify({
        'qualifications_identical': is_identical,
        'text_similarity_between_resumes': sim,
        'baseline': {
            'male': base_male,
            'female': base_female,
            'score_gap': round(abs(base_male['score'] - base_female['score']), 1)
        },
        'debiased': {
            'male': deb_male,
            'female': deb_female,
            'score_gap': round(abs(deb_male['score'] - deb_female['score']), 1)
        }
    })

@app.route('/api/train_debiaser', methods=['POST'])
def train_debiaser():
    _, history = train_pytorch_adversarial_model(epochs=40)
    return jsonify({'status': 'Success', 'training_history': history})

@app.route('/api/audit', methods=['GET'])
def get_audit_results():
    audit_data = run_fairness_audit()
    return jsonify(audit_data)

@app.route('/api/audit/download', methods=['GET'])
def download_audit_csv():
    filepath = "data/synthetic_candidates.csv"
    if not os.path.exists(filepath):
        generate_synthetic_candidate_dataset(filepath)
    return send_file(filepath, as_attachment=True, download_name="HireFair_Audit_Report.csv", mimetype="text/csv")

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)