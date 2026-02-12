Q1: Most Dangerous Failure Mode

The most dangerous failure mode of my system is false negatives, when the system fails to detect a patient who is actually deteriorating. My model had 53 false negatives (0.3% miss rate), with all of them in Patient 6 who had arrhythmia with irregular heartbeat patterns.

This is the most dangerous failure because:
1. A missed cardiac event means the paramedic has no early warning
2. Intervention may come too late without an alert
3. The risk of missing a heart attack because of reaching to the hospital in 5 to 20 minutes

While my system has excellent 98.9% recall, 1.1% miss rate presents an unacceptable level of risk because it could mean 1-2 preventable deaths per 100 critical patients. This is especially concerning for arrhythmia cases, where my model clearly struggles with irregular patterns that don't fit the normal deterioration profile. 

The consequence: A paramedic trusts the system, gets no alert, doesn't check the patient closely, and arrives at the hospital with a patient in cardiac arrest.This is far more dangerous than false positives (1,138 cases), because a false alarm just means a paramedic spends 30 seconds double-checking vitals where it can be life saving but little annoying.

Q2: How to Reduce False Alerts Without Missing Deterioration

My baseline model had a major problem: 2,060 false alarms (35.1% false positive rate). This creates "alert fatigue" where paramedics start ignoring alerts because most are false results.

I implemented two major improvements:

1. Temporal Smoothing (30-second window)
It made alerts for every 30 seconds intead on single bad reading, this filtered out:
- Road bump spikes (1-2 second transients)
- Sensor movement glitches
- Brief patient shifts

Reason I took 30 seconds:
- <30 seconds: Still too noisy and had lots of transient spikes
- >30 seconds: Delays real alerts
- 30 seconds: was perfect because it filters noise while real deterioration persists

2. Lower Contamination (0.1 → 0.05)
"I made the model more selective by adjusting the anomaly threshold (from 10% to 5%). This refined the criteria for what the system flags as abnormal."

Results:
- False alarms: 2,060 -> 1,138 (leading to 45% reduction) 
- Recall: 99.7% -> 98.9% (small drop) 
- Still detects events 5-19 minutes early 

19.4% false positives rate were taken as acceptable for following reasons
1. In a life-critical system, missing events is far more worse than false alarms
2. 45% reduction makes the system more trustworthy
3. Paramedics can verify in every 30 seconds (low cost)
4. Alternative would be lowering recall below 95%, which is unsafe

If I reduce false positives further it can lead to
- Delays if i increase smoothing window beyond 30 seconds
- Recall drops if I lower the contamination below 0.05
- Failure to detect gradual deterioration by using stricter thresholds

The current balance prioritizes safety while maintaining paramedic trust.

Q3: What Should Never Be Fully Automated

I believe the following should never be fully automated:

1. Medical Interventions
The system should never automatically:
- Administer the medication to the patient
- Perform defibrillation to the patient even if he/she is critical
- Change the dosages without consulating
- Call to the hospital without human internvention 

Why? Because my model has 1.1% error rate. Even one wrong decision like automated shock or drug dose could kill someone. Humans must always make the final decision on treatment.

2. Overriding Paramedic Judgment
The system should never:
- Prevent a paramedic from ignoring an alert(these are usually false positives) 
- takeover the manual control
- automatically execute clinical actions based on AI recommendation

Why? During development, I saw that Patient 6 who had arrhythmia triggered many false alarms due to naturally irregular heartbeat. An experienced paramedic would recognize this patient's baseline is irregular where as the AI doesn't have this input. The paramedic must have override authority on this

3. Sole Decision-Making Without Human Verification
The system should never be the only factor in decisions like:
- Whether to divert location to a different hospital
- Whether to attempt emergency procedures in the ambulance

Why? My system shows what I learned: AI sees patterns in numbers, but doesn't understand context. It doesn't know:
- Patient's medical history
- Current medications affecting the vitals
- Patient is awake and talking normally despite critical alert
- to diffrentiate Equipment malfunction and real problem

The Appropriate Role:
AI should be a decision support tool, not a decision maker:
- AI should give sign "Alert! High risk detected. Confidence: 85%"
- Paramedic Checks patient, uses judgment, decides action
- AI provides early warning, humans make final decision

This is critical because during this project I learned that even with 98.9% accuracy, the 1.1% of cases where I'm wrong could be fatal if there's no human oversight. Medical AI should support decision not authority, and it shouldn't replace.

**Author:** Spurthi Pattanashetti  
**Date:** February 12, 2026  
**Assignment:** Gray Mobility Smart Ambulance ML System