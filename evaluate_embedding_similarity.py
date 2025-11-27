#!/usr/bin/env python3

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

EXAMPLES = [
    {
        "transcript": "Customer called about billing issue. Their credit card was charged twice for the same service. Agent verified duplicate charge in system. Refund of $49.99 processed. Customer confirmed email address for confirmation.",
        "good_summary": "Customer reported duplicate charge of $49.99. Agent verified and processed refund.",
        "hallucinated_summary": "Customer complained about poor service quality and requested upgrade to premium plan at discounted rate. Agent approved 50% discount for 6 months."
    },
    {
        "transcript": "Customer wants to cancel internet service. Moving to new city next month. Agent confirmed cancellation date for March 15th. No early termination fee applies. Equipment return label will be emailed.",
        "good_summary": "Customer cancelling internet service due to relocation. Cancellation scheduled for March 15th, no fees, return label to be emailed.",
        "hallucinated_summary": "Customer extremely dissatisfied with internet speeds and frequent outages. Demanded compensation for service disruptions over past year. Agent offered $200 credit to account."
    },
    {
        "transcript": "Technical support call. Customer's router not connecting to internet. Agent walked through power cycle procedure. Checked customer's modem status lights. All lights green after restart. Connection restored successfully.",
        "good_summary": "Customer had router connectivity issue. Resolved through power cycle, connection restored.",
        "hallucinated_summary": "Customer's router hardware is defective and needs replacement. Agent authorized expedited shipping of new router at no charge. Arrival expected within 24 hours."
    },
    {
        "transcript": "Customer inquiring about data usage. Current month usage is 87GB out of 500GB limit. Agent explained overage charges are $10 per 50GB. Customer satisfied with explanation. No plan change requested.",
        "good_summary": "Customer checked data usage (87GB/500GB). Agent explained overage charges. No changes made.",
        "hallucinated_summary": "Customer exceeded data limit and incurred $150 in overage fees. Very upset about unexpected charges. Agent waived all fees and upgraded customer to unlimited data plan permanently."
    },
    {
        "transcript": "Customer called to update payment method. Old credit card expired. Agent securely collected new card details. Updated billing system. Confirmed next charge date is April 1st for $59.99 monthly plan.",
        "good_summary": "Customer updated expired credit card. Next billing date April 1st for $59.99.",
        "hallucinated_summary": "Customer disputed recent charges as fraudulent. Agent opened fraud investigation case. Temporarily suspended account and issued provisional credit of $300 pending investigation outcome."
    },
    {
        "transcript": "Testing one two three four five six seven eight nine ten.",
        "good_summary": "Audio test count from one to ten.",
        "hallucinated_summary": "Customer requested technical support for microphone issues. Agent scheduled on-site technician visit for hardware replacement."
    },
    {
        "transcript": "Hello this is customer service how can I help you today.",
        "good_summary": "Agent greeting customer.",
        "hallucinated_summary": "Customer called to file complaint about service interruption lasting three days. Agent provided compensation credit of $75 to customer account."
    },
    {
        "transcript": "Account number verified. Thank you.",
        "good_summary": "Account verification completed.",
        "hallucinated_summary": "Customer wants to upgrade service package to premium tier. Agent explained all benefits including unlimited data, priority support, and free equipment upgrades."
    }
]

model = SentenceTransformer('all-MiniLM-L6-v2')

print("Evaluating transcript-summary cosine similarity for hallucination detection\n")
print("=" * 80)

good_similarities = []
hallucinated_similarities = []

for i, example in enumerate(EXAMPLES, 1):
    print(f"\nExample {i}:")
    print(f"Transcript: {example['transcript'][:100]}...")

    transcript_embedding = model.encode([example['transcript']])
    good_embedding = model.encode([example['good_summary']])
    hallucinated_embedding = model.encode([example['hallucinated_summary']])

    good_sim = cosine_similarity(transcript_embedding, good_embedding)[0][0]
    hallucinated_sim = cosine_similarity(transcript_embedding, hallucinated_embedding)[0][0]

    good_similarities.append(good_sim)
    hallucinated_similarities.append(hallucinated_sim)

    print(f"\nGood summary similarity: {good_sim:.4f}")
    print(f"Good summary: {example['good_summary']}")
    print(f"\nHallucinated summary similarity: {hallucinated_sim:.4f}")
    print(f"Hallucinated summary: {example['hallucinated_summary']}")
    print(f"\nDifference: {good_sim - hallucinated_sim:.4f}")

print("\n" + "=" * 80)
print("\nAGGREGATE STATISTICS:")
print(f"\nGood summaries - Mean: {np.mean(good_similarities):.4f}, Std: {np.std(good_similarities):.4f}")
print(f"Min: {np.min(good_similarities):.4f}, Max: {np.max(good_similarities):.4f}")

print(f"\nHallucinated summaries - Mean: {np.mean(hallucinated_similarities):.4f}, Std: {np.std(hallucinated_similarities):.4f}")
print(f"Min: {np.min(hallucinated_similarities):.4f}, Max: {np.max(hallucinated_similarities):.4f}")

print(f"\nMean difference: {np.mean(good_similarities) - np.mean(hallucinated_similarities):.4f}")

print("\n" + "=" * 80)
print("\nEVALUATION:")

threshold_candidates = np.arange(0.3, 0.7, 0.05)
best_threshold = None
best_accuracy = 0

for threshold in threshold_candidates:
    good_correct = sum(1 for sim in good_similarities if sim >= threshold)
    hallucinated_correct = sum(1 for sim in hallucinated_similarities if sim < threshold)
    accuracy = (good_correct + hallucinated_correct) / (len(good_similarities) + len(hallucinated_similarities))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold

print(f"\nBest threshold: {best_threshold:.2f}")
print(f"Accuracy at threshold: {best_accuracy * 100:.1f}%")

good_correct = sum(1 for sim in good_similarities if sim >= best_threshold)
hallucinated_correct = sum(1 for sim in hallucinated_similarities if sim < best_threshold)

print(f"\nAt threshold {best_threshold:.2f}:")
print(f"Good summaries correctly accepted: {good_correct}/{len(good_similarities)}")
print(f"Hallucinated summaries correctly rejected: {hallucinated_correct}/{len(hallucinated_similarities)}")

overlap = (min(good_similarities) < best_threshold < max(good_similarities)) or \
          (min(hallucinated_similarities) < best_threshold < max(hallucinated_similarities))

if overlap:
    print("\nWARNING: Ranges overlap - some false positives/negatives are inevitable")
else:
    print("\nGOOD: Clean separation between good and hallucinated summaries")

print("\n" + "=" * 80)
print("\nCONCLUSION:")
print("This approach shows promise for detecting egregious hallucinations where")
print("summaries contain completely fabricated information unrelated to the transcript.")
print("\nHowever, it may struggle with:")
print("- Subtle hallucinations (small details changed)")
print("- Summaries that are accurate but use different terminology")
print("- Very short transcripts/summaries where embeddings are less reliable")
