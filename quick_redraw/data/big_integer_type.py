from sqlalchemy import BigInteger
from sqlalchemy.dialects import sqlite

# sqlite does not allow BigInteger as a primary key with autoincrement.  Use an integer for sqlite (local testing)
# but BigInteger elsewhere
BigIntegerType = BigInteger()
BigIntegerType = BigIntegerType.with_variant(sqlite.INTEGER(), 'sqlite')
