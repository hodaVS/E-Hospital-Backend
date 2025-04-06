# E-Hospital Prescription - Backend Testing Instructions

## Backend
## Heroku Link:  https://e-hospital-prescription-294a0e858fcd.herokuapp.com

### 1. /chat 
- **Method**: POST
- **URL**: `https://e-hospital-prescription-294a0e858fcd.herokuapp.com/chat`
- **Body**:   
  ```json{
   { "text" : "Patient: John Doe, Diagnosis: Fatty Liver. Warfarin 500mg capsules,  1 capsule by mouth three times daily for 10 days, 30 capsules, 0 Refills. Also, Ibuprofen 200mg ,
   1-2 tablets by mouth every 4-6 hours as needed for pain, 60 tablets, 2 refill. Rifaximin 550mg tablets Dosage: 1 tablet three times daily for 3 days.Quantity: 9 tablets, Refills:
  0 Note: Take Warfarin with food to avoid stomach upset."}
### 2. /transcribe_stream
- **URL**: `https://e-hospital-prescription-294a0e858fcd.herokuapp.com/transcribe_stream`

### 3. /save_prescription
- **URL**: `https://e-hospital-prescription-294a0e858fcd.herokuapp.com/save_prescription`


## Frontend 
You can interact with the backend through the frontend:

URL: (https://e-hospital-frontend-f30427acbeb1.herokuapp.com/)

Steps:
### Chat Mode
1. Open the link in a browser.

2. In the text input field, paste:

"Patient: John Doe, Diagnosis: Fatty Liver. Warfarin 500mg capsules, 1 capsule by mouth three times daily for 10 days, 30 capsules, 0 Refills. Also, Ibuprofen 200mg, 1-2 tablets by mouth every 4-6 hours as needed for pain, 60 tablets, 2 refill. Rifaximin 550mg tablets Dosage: 1 tablet three times daily for 3 days. Quantity: 9 tablets, Refills: 0 Note: Take Warfarin with food to avoid stomach upset."

3. Click "Send".
### Voice Mode
Steps:
1. Open the link in a browser.
2. Click the "Start Recording" button.
3. Allow microphone access when prompted by the browser.
3, Speak clearly into your microphone.
4. Click "Stop Recording".
5. Click "Play Audio" to review your recording.
6. Click "Submit Audio".


    
