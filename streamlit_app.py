# BEFORE RUNNING THIS SCRIPT:
# Ensure you install the following packages before running this script
# pip install streamlit transformers torch numpy

# Start the app with the following command:
# streamlit run streamlit_app.py

import string
import torch
import numpy as np
import streamlit as st
from transformers import GPT2LMHeadModel, RobertaTokenizerFast

# Constants
TOKENIZER_PATH = "./uniqpass-v16-passwords-tokenizer/"
MODEL_PATH = "./uniqpass-v16-passwords-trained/last"
MAXCHARS = 16
DEVICE = "cpu"

# Check if MPS is available and set it as the default device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

print(f"Using device: {DEVICE}")

# Initialize tokenizer and model
tokenizer = RobertaTokenizerFast.from_pretrained(
    TOKENIZER_PATH,
    max_len=MAXCHARS + 2,
    padding="max_length",
    truncation=True,
    do_lower_case=False,
    strip_accents=False,
    mask_token="<mask>",
    unk_token="<unk>",
    pad_token="<pad>",
    truncation_side="right",
)

model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).eval().to(DEVICE)


def get_tokens(tokenizer, symbols):
    return tokenizer(symbols, add_special_tokens=False).input_ids


def create_token_dict(tokenizer):
    lowercase = list(string.ascii_lowercase)
    uppercase = list(string.ascii_uppercase)
    digits = list(string.digits)
    punctuation = list(string.punctuation)

    lowercase_tokens = get_tokens(tokenizer, lowercase)
    uppercase_tokens = get_tokens(tokenizer, uppercase)
    digits_tokens = get_tokens(tokenizer, digits)
    punctuation_tokens = get_tokens(tokenizer, punctuation)

    return {
        "l": lowercase_tokens,
        "u": uppercase_tokens,
        "d": digits_tokens,
        "p": punctuation_tokens,
    }


def generate_password(template, num_generations=1):
    generated = 0
    generations = []

    while generated < num_generations:
        generation = torch.tensor([tokenizer.bos_token_id]).unsqueeze(0)
        current_length = 1

        for char in template:
            if char in token_dict:
                bad_tokens = [i for i in all_tokens if i not in token_dict[char]]
            else:
                bad_tokens = [[tokenizer.eos_token_id]]

            generation = model.generate(
                generation.to(DEVICE),
                do_sample=True,
                max_length=current_length + 1,
                pad_token_id=tokenizer.pad_token_id,
                num_return_sequences=1,
                bad_words_ids=bad_tokens,
            )
            current_length += 1

        if not 2 in generation.flatten():
            generations.append(generation)
            generated += 1

    return torch.cat(generations, 0)[:, 1:]


def calculate_password_probability(model, tokenizer, password, device):
    # Tokenize the password and convert to tensor format
    input_ids = tokenizer.encode(password, return_tensors="pt").to(device)

    # Get the model's logit predictions for the password
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Convert logits to log probabilities
    log_probabilities = torch.nn.functional.log_softmax(logits, dim=-1)

    # Gather the log probabilities of the actual tokens
    selected_log_probs = (
        log_probabilities[:, :-1].gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    )

    # Sum the log probabilities
    total_log_probability = selected_log_probs.sum().item()

    return total_log_probability


# Create Sample set of passwords for demo
token_dict = create_token_dict(tokenizer)

all_tokens = [[i] for i in range(len(tokenizer))]

generations = generate_password("uu**p*dd", 3)

decoded_passwords = tokenizer.batch_decode(generations)

# Add some common passwords to the list to ensure known results rank higher in log likelihood
decoded_passwords.append("password")
decoded_passwords.append("password")
decoded_passwords.append("123456")
decoded_passwords.append("qwerty")
# decoded_passwords.append("abc123")
# decoded_passwords.append("password1")
# decoded_passwords.append("password123")
# decoded_passwords.append("123456789")
# decoded_passwords.append("12345678")
# decoded_passwords.append("12345")
# decoded_passwords.append("iloveyou")
# decoded_passwords.append("111111")

log_likelihoods = []
for i, password in enumerate(decoded_passwords):
    password_probability = calculate_password_probability(
        model, tokenizer, password, DEVICE
    )
    log_likelihoods.append(password_probability)
    print(
        f"Generated Password {i+1}: {password} - Log Likelihood: {password_probability}"
    )


# Function to convert log likelihood to a strength percentage
def log_likelihood_to_strength_percentage(log_likelihood):
    min_log_likelihood = -45  # corresponds to 100%
    max_log_likelihood = -5  # corresponds to 0%

    # Ensure the log likelihood is within the range we're considering
    log_likelihood = max(min_log_likelihood, min(max_log_likelihood, log_likelihood))

    # Calculate the percentage
    strength_percentage = (
        1
        - (log_likelihood - min_log_likelihood)
        / (max_log_likelihood - min_log_likelihood)
    ) * 100

    return strength_percentage


# Collect strength ratings for all passwords
overall_strength = np.mean(log_likelihoods)
strengths = [log_likelihood_to_strength_percentage(ll) for ll in log_likelihoods]

# Calculate the overall strength as an average of individual strengths
overall_strength = sum(strengths) / len(strengths)


def get_strength_color(percentage):
    if percentage >= 70:
        return "green"
    elif percentage >= 40:
        return "orange"
    else:
        return "red"


def get_strength_label(percentage):
    if percentage >= 70:
        return "Strong"
    elif percentage >= 40:
        return "Medium"
    else:
        return "Weak"


# Custom Progress bar to support color coding
def CreateProgressBar(pg_caption, pg_int_percentage, pg_colour, pg_bgcolour):
    pg_int_percentage = str(pg_int_percentage).zfill(2)
    pg_html = f"""<table style="width:100%; border-style: none;">
                        <tr style='font-weight:bold;'>
                            <td style='background-color:{pg_bgcolour};'>{pg_caption}: <span style='accent-color: {pg_colour}; bgcolor: transparent;'>
                                <progress value='{pg_int_percentage}' max='100'>{pg_int_percentage}%</progress> </span>{pg_int_percentage}% 
                            </td>
                        </tr>
                    </table><br>"""
    return pg_html


# Streamlit app layout in two columns format
col1, col2 = st.columns(2)

with col1:
    st.header("Security Audit")
    security_score = overall_strength
    st.metric(label="", value=f"{security_score}%", delta="Strong")
    st.progress(security_score / 100)

    st.subheader("All")
    st.text(len(decoded_passwords))

    st.subheader("Reused")
    # check decoded_passwords for reused passwords
    st.text((len(decoded_passwords) - len(set(decoded_passwords))))

    st.subheader("Weak")
    # Check for weak passwords with log likelihoods > -20
    st.text((len([i for i in log_likelihoods if i > -20])))


with col2:
    st.header("Name")

    # loop through decoded_passwords and log_likelihoods to get the strength of each account
    for account, strength in zip(decoded_passwords, strengths):
        st.text(account)

        # color code the strength of each account based on the strength percentage use green, orange and red
        color = get_strength_color(strength)
        label = get_strength_label(strength)
        # print(account, strength, color)

        st.markdown(CreateProgressBar(label, strength, color, "transparent"), True)
