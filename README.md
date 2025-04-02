# E-Hospital Prescription Backend Testing Instructions
## Prerequisites
- Postman installed (download from https://www.postman.com/downloads/).
- An audio file (e.g., `.wav`) for testing `/transcribe_stream`.

1. Import the Postman Collection:
   - Open Postman.
   - Click "Import" in the top left.
   - Select the file `E-Hospital-Prescription-Backend.postman_collection.json` and import it.
2. You’ll see a collection named "E-Hospital Prescription Backend" with three requests.

## Testing the Endpoints
### 1. /chat 
- **Method**: POST
- **URL**: `https://e-hospital-prescription-294a0e858fcd.herokuapp.com/chat`
- **Body**:   
  ```json{
    "text" : "Patient: John Doe, Diagnosis: Fatty Liver. Warfarin 500mg capsules,  1 capsule by mouth three times daily for 10 days, 30 capsules, 0 Refills. Also, Ibuprofen 200mg , 1-2 tablets by mouth every 4-6 hours as needed for pain, 60 tablets, 2 refill. Rifaximin 550mg tablets Dosage: 1 tablet three times daily for 3 days.Quantity: 9 tablets, Refills: 0 Note: Take Warfarin with food to avoid stomach upset."
}
