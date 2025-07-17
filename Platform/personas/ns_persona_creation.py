import pandas as pd
import time
import google.generativeai as genai
import os

# === SETUP YOUR API KEY ===
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# === Initialize Gemini Flash 2.0 ===
model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")

# === Load Excel file ===
input_file = "../../personas/tweets_replies_disaggr_1st_iter.xlsx"
init = 1068
df = pd.read_excel(input_file)[init:]

num_neutral = 0

# === Store outputs ===
output_data = []

# === Iterate through each row ===
for idx, row in df.iterrows():
    if num_neutral >= 1:
        print(f'last index: {idx}')
        break
    text = row["text"]
    try:
        #print(text)
        # PROMPT 1: Persona description
        prompt = f"""{text}
        Describe the person who wrote the previous message. Do it directly in the second person singular. Do not refer to the message itself. Be assertive and avoid uncertain language such as 'likely', 'seems', or 'probably'. Use confident and descriptive phrasing."""

        response = model.generate_content(prompt)
        persona_description = response.text.strip()

        print(persona_description)

        # PROMPT 2: Political standpoint
        prompt = f"""{text}
        Determine the political standpoint of who wrote the text. Just return either Democrat, Republican, or Neutral."""

        response = model.generate_content(prompt)
        political_standpoint = response.text.strip()

        print(political_standpoint)

        if "neutral" in political_standpoint.lower():
            num_neutral += 1

        # Save to output
        output_data.append({
            "persona_description": persona_description,
            "political_standpoint": political_standpoint
        })

    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        output_data.append({
            "persona_description": "Error",
            "political_standpoint": "Error"
        })
        break

# === Save output to CSV ===
output_df = pd.DataFrame(output_data)
output_df.to_csv(f"persona_descriptions_{init}.csv", index=False)
