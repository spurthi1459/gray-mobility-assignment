# Design Assumptions and Limitations

## Data Generation Assumptions

### Patient Scenarios
1. **Normal Transport (Patients 1-2)**
   - Baseline HR: 72-80 bpm (age-adjusted)
   - Baseline SpO2: 96-99%
   - Baseline BP: 118-130 / 78-85 mmHg
   - Assumption: Stable patients with minor physiological variation only

2. **Deterioration - Shock (Patient 3)**
   - Gradual decline over 30 minutes
   - HR increases to ~115 bpm
   - SpO2 drops to ~89%
   - BP drops to 90/60 mmHg
   - Assumption: Hemorrhagic shock progression model

3. **Deterioration - Respiratory (Patient 4)**
   - Primary SpO2 decline (to ~85%)
   - Compensatory HR increase
   - Assumption: Progressive hypoxia (COPD, pneumonia)

4. **Acute Cardiac Event (Patient 5)**
   - Event occurs randomly between 5-20 minutes
   - Sudden tachycardia (140+ bpm)
   - SpO2 and BP drop
   - Assumption: Acute MI or arrhythmia onset

5. **Arrhythmia (Patient 6)**
   - Irregular HR with high variability
   - Assumption: Atrial fibrillation pattern

### Artifact Assumptions
1. **Road bumps:** 5-15 per 30-minute transport
2. **Motion artifacts:** Affect HR and SpO2 simultaneously
3. **Sensor dropouts:** 1-3% of readings
4. **Artifact magnitude:** 
   - HR spikes: ±10-15 bpm
   - SpO2 drops: -2 to -5%

---

## Model Training Assumptions

### Feature Engineering
1. **60-second rolling window**: Chosen based on clinical practice
   - Vitals are typically assessed over 1-minute intervals
   - Provides balance between responsiveness and noise reduction

2. **Z-score calculation**: Uses entire patient dataset
   - Assumes patient's own baseline is most relevant
   - May not generalize to different patient populations

### Anomaly Detection
1. **Contamination rate: 0.05** (5% expected anomalies)
   - Based on empirical testing
   - Assumes ~5% of normal transport data has transient abnormalities

2. **Isolation Forest choice:**
   - No labeled data available (unsupervised learning required)
   - Assumes anomalies are "few and different"
   - May not detect slow, gradual changes well

3. **Temporal smoothing (30 seconds):**
   - Requires sustained anomaly (not just 1-second spike)
   - Assumption: True deterioration persists for >30 seconds
   - May delay detection of very sudden events

---

## Risk Scoring Assumptions

### Weight Distribution
- Vital severity: 40%
- Trend direction: 30%
- Anomaly confidence: 20%
- Vital combination: 10%

**Justification:**
- Current vitals most important (immediate danger)
- Trends indicate trajectory (early warning)
- ML confidence provides context
- Multiple abnormalities compound risk

### Risk Thresholds
- LOW: 0-3
- MODERATE: 4-6
- HIGH: 7-8
- CRITICAL: 9-10

**Assumption:** These thresholds would be validated with clinical data in production

---

## Clinical Assumptions

### Normal Ranges (Used for Severity Scoring)
- **Heart Rate:** 60-100 bpm (adults at rest)
- **SpO2:** >95% (normal), <90% (critical)
- **Systolic BP:** 90-140 mmHg
- **Diastolic BP:** 60-90 mmHg
- **MAP:** >70 mmHg (adequate perfusion)
- **Shock Index:** 0.5-0.7 (normal), >1.0 (shock)

**Limitation:** These are population averages; individual baselines vary

### Medical Indicators
1. **Shock Index (HR/SBP):** 
   - Assumption: Valid for adult patients
   - May not apply to children, athletes, elderly

2. **Mean Arterial Pressure:** 
   - Formula: (SBP + 2×DBP) / 3
   - Assumption: Standard clinical calculation

---

## API Assumptions

### Real-Time Predictions
1. **Single-point predictions have limited accuracy**
   - Missing: historical context, trends, baseline
   - Recommendation: Use batch endpoint with ≥5 minutes history

2. **Latency acceptable:** <100ms per prediction
   - Assumption: Ambulance telemetry can tolerate this delay

3. **No authentication/encryption:** 
   - Prototype only
   - Production would require HTTPS, authentication, HIPAA compliance

---

## Data Limitations

### Not Modeled
1. **Patient demographics:** Age, gender, medical history
2. **Medication effects:** Beta-blockers, vasopressors
3. **Environmental factors:** Temperature, altitude
4. **Movement artifacts from CPR**
5. **Multiple simultaneous patients**
6. **Equipment failures:** Battery, connectivity

### Synthetic Data Limitations
1. **No real physiological coupling** between vitals
   - Example: BP and HR relationship oversimplified
2. **Deterministic noise patterns**
   - Real artifacts more chaotic
3. **No chronic conditions** modeled (diabetes, heart failure)

---

## Evaluation Assumptions

### Ground Truth Labels
1. **Normal transport = Normal (label: 0)**
2. **Deterioration/Acute = Anomaly (label: 1)**

**Exception:** First 5 minutes of deterioration labeled as normal
- Rationale: Early deterioration may be clinically indistinguishable

### Metrics Interpretation
1. **Recall prioritized over Precision**
   - Assumption: Missing critical events is worse than false alarms
   - Trade-off accepted: 98.9% recall vs 19.4% FPR

2. **Alert latency measured from first true anomaly**
   - Assumption: Clinical deterioration has identifiable onset
   - Reality: Often gradual with no clear start time

---

## Known Limitations

### Model Limitations
1. **No concept drift detection**
   - Model may degrade if patient populations change
2. **No interpretability**
   - Isolation Forest is black-box
   - Cannot explain why specific prediction was made
3. **Fixed thresholds**
   - No adaptive learning from feedback

### System Limitations
1. **No redundancy or failover**
2. **No data validation** beyond range checks
3. **No audit logging** of predictions
4. **No performance monitoring** in production

---

## Future Validation Needed

Before clinical deployment:

1. **Validate on real patient data** from ambulances
2. **Clinical expert review** of alerts
3. **Prospective study** comparing to paramedic assessments
4. **Regulatory approval** (FDA clearance if in US)
5. **Integration testing** with ambulance telemetry systems
6. **Failure mode analysis** (FMEA)
7. **Privacy impact assessment** (HIPAA compliance)

---

## Ethical Considerations

1. **False negatives are dangerous** - system must never be sole decision-maker
2. **Alert fatigue** - too many false alarms reduce trust
3. **Bias potential** - trained only on synthetic data
4. **Liability** - who is responsible for missed detections?

**Recommendation:** Use as decision support only, not autonomous diagnosis