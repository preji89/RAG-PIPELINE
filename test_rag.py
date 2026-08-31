from rag_pipeline import RAGPipeline
from claude_rag_pipeline import ClaudeRAGPipeline

QUESTION = "Can I get my money back if I don't like the product?"

# Initialize (both pipelines share the same "chunks" table)
rag = RAGPipeline()
claude_rag = ClaudeRAGPipeline()

# Ingest documents once
rag.ingest_document("""
Our refund policy: We offer a 14-day money-back guarantee for all annual plans.
Monthly plans can be cancelled anytime but are not refundable.
To request a refund, email support@example.com with your order ID.
""", source="refund-policy.md")

rag.ingest_document("""
Pricing Plans:
- Starter: $9/month or $90/year (save $18)
- Pro: $29/month or $290/year (save $58)
- Enterprise: Custom pricing, contact sales
All plans include 14-day free trial.
""", source="pricing.md")

# Query with the OpenAI-generation pipeline
print("=== OpenAI (gpt-4o-mini) ===")
print(rag.query(QUESTION))

# Query with the Claude-generation pipeline
print("\n=== Claude (claude-opus-5) ===")
print(claude_rag.query(QUESTION))

# Expected output (both):
# Yes, you can get a refund within 14 days for annual plans.
# Monthly plans are not refundable but can be cancelled anytime.
# Email support@example.com with your order ID to request a refund.
#
# Sources: refund-policy.md
