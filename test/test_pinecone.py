from pinecone import Pinecone

pc = Pinecone(api_key="pcsk_2yjVuy_3zWV39ZwNeBiKmiTdPEvgzz772cAuDLYGE8JVwhn9FoPfhcB5xRgfToLrxPEPrx")
indexes = pc.list_indexes()
print(indexes)