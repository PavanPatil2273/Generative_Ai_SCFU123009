from langchain_core.tools import tool
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

orders = {
    "ORD-4521": "Shipped - arriving tomorrow",
    "ORD-1234": "Delivered",
    "ORD-7890": "Processing"
}


@tool
def get_order_status(order_id: str) -> str:
    """Get the shipping status of an order."""
    return orders.get(order_id, "Order not found")


model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

model_with_tool = model.bind_tools([get_order_status])


question = input("Ask: ")

response = model_with_tool.invoke(question)

print("Model response:", response)
print("Tool calls:", response.tool_calls)