from langchain_core.output_parsers import CommaSeparatedListOutputParser

# 字串轉成list
parser = CommaSeparatedListOutputParser()
items = parser.parse("apple, banana, cherry, date")

print(f"Parsed items: {items}")
print(type(items))