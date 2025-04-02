from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
from openai import OpenAI
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

class PrescriptionBackend:
    def __init__(self):
        self.conversation_history = [
            {
                "role": "system",
                "content": "You are a helpful assistant that generates prescriptions. Always return the prescription in the following JSON format: (Warn doctor in Description if you suspect any drug conflicts). If any information is missing, use 'None' as the value for that field."
                           "{ "
                           "\"Prescriptions\": [ "
                           "{ "
                           "\"DiagnosisInformation\": { \"Diagnosis\": \"<diagnosis>\", \"Medicine\": \"<medicine>\" }, "
                           "\"MedicationDetails\": { "
                           "\"Dose\": \"<dose>\", "
                           "\"DoseUnit\": \"<dose unit>\", \"DoseRoute\": \"<dose route>\", "
                           "\"Frequency\": \"<frequency>\", \"FrequencyDuration\": \"<frequency duration>\", "
                           "\"FrequencyUnit\": \"<frequency unit>\", \"Quantity\": \"<quantity>\", "
                           "\"QuantityUnit\": \"<quantity unit>\", \"Refill\": \"<refill>\", "
                           "\"Pharmacy\": \"<pharmacy>\" "
                           "}, "
                           "\"Description\": \"<description>\" "
                           "} ] "
                           "}"
            }
        ]

    def transcribe_audio(self, file_path):
        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcript.text

    def chat_with_gpt(self, prompt, messages=None, model="gpt-4"):
        if messages is None:
            messages = self.conversation_history.copy()
            
        messages.append({"role": "user", "content": prompt})

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=5000,
            temperature=0
        )
        gpt_response = completion.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": gpt_response})

        return gpt_response

    def process_transcription_request(self, audio_file):
        if not audio_file:
            logger.error("No audio file provided")
            return {"error": "No audio file provided", "logs": ["No audio file provided"]}, 400

        logs = [f"Received audio file: {audio_file.filename}"]
        
        try:
            temp_path = "temp_audio.wav"
            audio_file.save(temp_path)
            logs.append("Audio file saved to temp path")

            audio_file.seek(0)

            logger.info("Attempting Whisper transcription")
            with open(temp_path, "rb") as f:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f
                )
            user_input = transcript.text
            logs.append(f"Transcribed text: {user_input}")

            os.remove(temp_path)
            logs.append("Temporary file removed")

            system_message = {
                "role": "system",
                "content": "You are a helpful assistant that generates prescriptions. Always return the prescription in the following JSON format: (Warn doctor in Description if you suspect any drug conflicts). If any information is missing, use 'None' as the value for that field."
                           "{ \"Prescriptions\": [ { \"DiagnosisInformation\": { \"Diagnosis\": \"<diagnosis>\", \"Medicine\": \"<medicine>\" }, \"MedicationDetails\": { \"Dose\": \"<dose>\", \"DoseUnit\": \"<dose unit>\", \"DoseRoute\": \"<dose route>\", \"Frequency\": \"<frequency>\", \"FrequencyDuration\": \"<frequency duration>\", \"FrequencyUnit\": \"<frequency unit>\", \"Quantity\": \"<quantity>\", \"QuantityUnit\": \"<quantity unit>\", \"Refill\": \"<refill>\", \"Pharmacy\": \"<pharmacy>\" }, \"Description\": \"<description>\" } ] }"
            }

            logger.info("Requesting GPT-4 completion")
            completion = client.chat.completions.create(
                model="gpt-4",
                messages=[system_message, {"role": "user", "content": user_input}],
                max_tokens=500,
                temperature=0.1
            )

            gpt_response = completion.choices[0].message.content.strip()
            logs.append(f"GPT-4 response: {gpt_response}")

            try:
                gpt_response = gpt_response.replace('1-2', '"1-2"')
                prescription = json.loads(gpt_response)
                logs.append("GPT-4 response parsed as JSON")
                
                for p in prescription.get("Prescriptions", []):
                    p.setdefault("DiagnosisInformation", {"Diagnosis": None, "Medicine": None})
                    p.setdefault("MedicationDetails", {
                        "Dose": None, "DoseUnit": None, "DoseRoute": None,
                        "Frequency": None, "FrequencyDuration": None, "FrequencyUnit": None,
                        "Quantity": None, "QuantityUnit": None, "Refill": None, "Pharmacy": None
                    })
                    p.setdefault("Description", None)

                return {
                    "response": prescription,
                    "transcript": user_input,
                    "logs": logs
                }, 200

            except json.JSONDecodeError as e:
                logs.append(f"JSON parsing failed: {str(e)}")
                return {
                    "error": "Failed to generate prescription",
                    "transcript": user_input,
                    "details": str(e),
                    "logs": logs
                }, 500

        except Exception as e:
            logs.append(f"Audio processing failed: {str(e)}")
            return {
                "error": "Audio processing failed",
                "details": str(e),
                "logs": logs
            }, 500

    def process_chat_request(self, user_input):
        if not user_input:
            return {"error": "No text provided"}, 400

        try:
            system_message = {
                "role": "system",
                "content": "You are a helpful assistant that generates prescriptions. Always return the prescription in the following JSON format: (Warn doctor in Description if you suspect any drug conflicts). If any information is missing, use 'None' as the value for that field."
                        "{ \"Prescriptions\": [ { \"DiagnosisInformation\": { \"Diagnosis\": \"<diagnosis>\", \"Medicine\": \"<medicine>\" }, \"MedicationDetails\": { \"Dose\": \"<dose>\", \"DoseUnit\": \"<dose unit>\", \"DoseRoute\": \"<dose route>\", \"Frequency\": \"<frequency>\", \"FrequencyDuration\": \"<frequency duration>\", \"FrequencyUnit\": \"<frequency unit>\", \"Quantity\": \"<quantity>\", \"QuantityUnit\": \"<quantity unit>\", \"Refill\": \"<refill>\", \"Pharmacy\": \"<pharmacy>\" }, \"Description\": \"<description>\" } ] }"
            }

            completion = client.chat.completions.create(
                model="gpt-4",
                messages=[system_message, {"role": "user", "content": user_input}],
                max_tokens=500,
                temperature=0.1
            )

            gpt_response = completion.choices[0].message.content.strip()
            gpt_response = gpt_response.replace('1-2', '"1-2"')

            if not gpt_response.strip().endswith("}"):
                return {
                    "response": {
                        "Prescriptions": [
                            {
                                "DiagnosisInformation": {"Diagnosis": None, "Medicine": None},
                                "MedicationDetails": {
                                    "Dose": None, "DoseUnit": None, "DoseRoute": None,
                                    "Frequency": None, "FrequencyDuration": None, "FrequencyUnit": None,
                                    "Quantity": None, "QuantityUnit": None, "Refill": None, "Pharmacy": None
                                },
                                "Description": "Please try again with proper prescription content."
                            }
                        ]
                    }
                }, 200

            try:
                prescription = json.loads(gpt_response)
                for p in prescription.get("Prescriptions", []):
                    p.setdefault("DiagnosisInformation", {"Diagnosis": None, "Medicine": None})
                    p.setdefault("MedicationDetails", {
                        "Dose": None, "DoseUnit": None, "DoseRoute": None,
                        "Frequency": None, "FrequencyDuration": None, "FrequencyUnit": None,
                        "Quantity": None, "QuantityUnit": None, "Refill": None, "Pharmacy": None
                    })
                    p.setdefault("Description", None)

                return {"response": prescription}, 200
            except json.JSONDecodeError:
                return {
                    "response": {
                        "Prescriptions": [
                            {
                                "DiagnosisInformation": {"Diagnosis": None, "Medicine": None},
                                "MedicationDetails": {
                                    "Dose": None, "DoseUnit": None, "DoseRoute": None,
                                    "Frequency": None, "FrequencyDuration": None, "FrequencyUnit": None,
                                    "Quantity": None, "QuantityUnit": None, "Refill": None, "Pharmacy": None
                                },
                                "Description": "Please try again with proper prescription content."
                            }
                        ]
                    }
                }, 200

        except Exception as e:
            return {
                "response": {
                    "Prescriptions": [
                        {
                            "DiagnosisInformation": {"Diagnosis": None, "Medicine": None},
                            "MedicationDetails": {
                                "Dose": None, "DoseUnit": None, "DoseRoute": None,
                                "Frequency": None, "FrequencyDuration": None, "FrequencyUnit": None,
                                "Quantity": None, "QuantityUnit": None, "Refill": None, "Pharmacy": None
                            },
                            "Description": "Please try again with proper prescription content."
                        }
                    ]
                }
            }, 200

    def save_prescription_data(self, prescription_data):
        if not prescription_data or 'prescription' not in prescription_data:
            logger.error("No prescription data provided")
            return {"error": "No prescription data provided"}, 400

        prescription = prescription_data['prescription']
        save_file = "/data/prescriptions_dataset.json" if os.getenv("RENDER") else "prescriptions_dataset.json"
        
        try:
            prescriptions = []
            if os.path.exists(save_file):
                with open(save_file, 'r') as f:
                    try:
                        prescriptions = json.load(f)
                    except json.JSONDecodeError:
                        prescriptions = []

            from datetime import datetime
            prescription_entry = {
                "prescription": prescription,
                "timestamp": datetime.now().isoformat()
            }
            prescriptions.append(prescription_entry)

            with open(save_file, 'w') as f:
                json.dump(prescriptions, f, indent=2)

            logger.info(f"Prescription saved successfully: {prescription}")
            return {"message": f"Prescription for {prescription['DiagnosisInformation']['Medicine']} saved successfully"}, 200

        except Exception as e:
            logger.error(f"Failed to save prescription: {str(e)}")
            return {"error": "Failed to save prescription", "details": str(e)}, 500
