# Error analysis (test, finetuned, n=206)

## Missing information read as a negative finding — 28 cases

- **#169** (sigir-201410, NCT02097186, exclusion): *Patients with an estimated pre-operative glomerular filtration rate < 30mls/min/1.73m2*  
  expert **not enough information** [] · predicted **not excluded** []  
  expert evidence: —  
  predicted evidence: —

- **#174** (sigir-201410, NCT02097186, exclusion): *Patients with previous axillary surgery*  
  expert **not enough information** [] · predicted **not excluded** []  
  expert evidence: —  
  predicted evidence: —

## Decided a criterion the experts judged undecidable — 1 cases

- **#678** (sigir-20158, NCT02562040, inclusion): *Diagnosis of mild sleep-disordered breathing (MSDB) defined as meeting all of the following criteria: Caregiver report of habitual snoring that occurs most of the night on at least three nights per week, and has been present for at least three months (on average occurring > 3 nights per week or more half of sleep time)*  
  expert **not enough information** [0] · predicted **included** [0]  
  expert evidence: {0: 'A 10 yo boy with nighttime snoring, pauses in breathing, and restlessness with nighttime awakenings.'}  
  predicted evidence: {0: 'A 10 yo boy with nighttime snoring, pauses in breathing, and restlessness with nighttime awakenings.'}

## Cited evidence yet answered 'not enough information' — 6 cases

- **#161** (sigir-201410, NCT02097186, inclusion): *Patients undergoing elective carotid endarterectomy*  
  expert **not included** [] · predicted **not enough information** [0, 1, 2]  
  expert evidence: —  
  predicted evidence: {0: 'A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.', 1: 'She is now complaining of a cool right foot.', 2: 'Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.'}

- **#162** (sigir-201410, NCT02097186, inclusion): *Patients undergoing open abdominal aortic aneurysm repair*  
  expert **not included** [] · predicted **not enough information** [0, 1, 2]  
  expert evidence: —  
  predicted evidence: {0: 'A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.', 1: 'She is now complaining of a cool right foot.', 2: 'Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.'}

## Answered 'not enough information' although the note decides the criterion — 5 cases

- **#176** (sigir-201410, NCT02264964, inclusion): *No medical history of central vena catheterization.*  
  expert **included** [0] · predicted **not enough information** []  
  expert evidence: {0: 'A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.'}  
  predicted evidence: —

- **#177** (sigir-201410, NCT02264964, inclusion): *Maintenance hemodialysis after central vena catheterization.*  
  expert **not included** [] · predicted **not enough information** []  
  expert evidence: —  
  predicted evidence: —

## Missed a 'not applicable' criterion — 11 cases

- **#422** (sigir-201422, NCT00236912, inclusion): *If female, using birth control*  
  expert **not applicable** [3] · predicted **not enough information** []  
  expert evidence: {3: 'She has had no sexual partners and her menses are regular.'}  
  predicted evidence: —

- **#683** (sigir-20158, NCT02562040, exclusion): *Previous tonsillectomy, including partial tonsillectomy*  
  expert **not applicable** [] · predicted **not excluded** []  
  expert evidence: —  
  predicted evidence: —

## Opposite decision (met vs not met) — 1 cases

- **#164** (sigir-201410, NCT02097186, inclusion): *Patients undergoing surgical lower limb revascularisation (suprainguinal or infrainguinal)*  
  expert **not included** [] · predicted **included** [0, 2]  
  expert evidence: —  
  predicted evidence: {0: 'A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.', 2: 'Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.'}

## Other label error — 1 cases

- **#322** (sigir-201417, NCT01745731, exclusion): *Pregnant women or women of childbearing age who do not have adequate contraception.*  
  expert **not excluded** [0] · predicted **not applicable** []  
  expert evidence: {0: 'A 48-year-old white male with history of common variable immunodeficiency (CVID) with acute abdominal pain, fever, dehydration, HR of 132 bpm, BP 80/40.'}  
  predicted evidence: —

## Correct label but no evidence cited where experts cited some — 13 cases

- **#179** (sigir-201410, NCT02264964, exclusion): *Had been performed central venous puncture or catheterization before.*  
  expert **not excluded** [0] · predicted **not excluded** []  
  expert evidence: {0: 'A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.'}  
  predicted evidence: —

- **#196** (sigir-201411, NCT01723137, exclusion): *Less than 18 years of age*  
  expert **not excluded** [0] · predicted **not excluded** []  
  expert evidence: {0: 'A 40-year-old woman with no past medical history presents to the ER with excruciating pain in her right arm that had started 1 hour prior to her admission.'}  
  predicted evidence: —

## Correct label but cited sentences the experts did not — 15 cases

- **#307** (sigir-201417, NCT01745731, inclusion): *Patient selection should be cautious, covering basically 5 types of liver damage which must be submitted prior to liver volumetry: Metastatic Disease subsidiary right hepatectomy extended to segment IV Metastatic Disease subsidiary right hepatectomy with suspected diseased liver (neoadjuvant chemotherapy) (in cases of doubt may be used liver function test indocyanine green) Bilobar liver metastases with multiple nodules in the right lobe and more than 3 nodules greater than 30 mm in the left hepatic lobe (LHI) will perform lumpectomies the LHI + right portal branch ligation (or postoperative percutaneous embolization) in order to make right hepatectomy 4-6 weeks (two stage surgery) Subsidiary Hepatocarcinoma extended right hepatectomy Liver Injury benign / malignant (Hemangiomas, hydatid cysts or liver tumors / primary bile hepatoblastoma), which by extension threatens the viability of the remaining liver tissue.*  
  expert **not enough information** [3] · predicted **not enough information** [0, 1, 2, 3, 4]  
  expert evidence: {3: 'Exploratory laparotomy reveals a ruptured liver abscess, which is then surgically drained.'}  
  predicted evidence: {0: 'A 48-year-old white male with history of common variable immunodeficiency (CVID) with acute abdominal pain, fever, dehydration, HR of 132 bpm, BP 80/40.', 1: 'The physical examination is remarkable for tenderness and positive Murphy sign.', 2: 'Abdominal ultrasound shows hepatomegaly and abundant free intraperitoneal fluid.', 3: 'Exploratory laparotomy reveals a ruptured liver abscess, which is then surgically drained.', 4: 'After surgery, the patient is taken to the ICU.'}

- **#325** (sigir-201417, NCT02556359, inclusion): *Genetic patients*  
  expert **not enough information** [] · predicted **not enough information** [0]  
  expert evidence: —  
  predicted evidence: {0: 'A 48-year-old white male with history of common variable immunodeficiency (CVID) with acute abdominal pain, fever, dehydration, HR of 132 bpm, BP 80/40.'}

## Correct label but missed some expert evidence sentences — 2 cases

- **#420** (sigir-201422, NCT00236912, inclusion): *Two or more symptoms of acute appendicitis for at least 24 hours or radiologic evidence of complicated appendicitis*  
  expert **included** [0, 1, 2, 5] · predicted **included** [0, 1, 5]  
  expert evidence: {0: 'A 15-year-old girl presents to the ER with abdominal pain.', 1: 'The pain appeared gradually and was periumbilical at first, localizing to the right lower quadrant over hours.', 2: 'She has had no appetite since yesterday but denies diarrhea.', 5: 'On an abdominal ultrasound, a markedly edematous appendix is seen.'}  
  predicted evidence: {0: 'A 15-year-old girl presents to the ER with abdominal pain.', 1: 'The pain appeared gradually and was periumbilical at first, localizing to the right lower quadrant over hours.', 5: 'On an abdominal ultrasound, a markedly edematous appendix is seen.'}

- **#691** (sigir-20158, NCT00393913, inclusion): *Experiences symptoms of OSA, including snoring and sleepiness*  
  expert **included** [0, 2] · predicted **included** [0]  
  expert evidence: {0: 'A 10 yo boy with nighttime snoring, pauses in breathing, and restlessness with nighttime awakenings.', 2: "The boy's teacher recently contacted his parents because she was concerned about his declining grades, lack of attention, and excessive sleepiness during class."}  
  predicted evidence: {0: 'A 10 yo boy with nighttime snoring, pauses in breathing, and restlessness with nighttime awakenings.'}
