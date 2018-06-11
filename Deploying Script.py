import os
import re
import un_model_oneHot as model
from datetime import datetime
from pymongo import MongoClient, ASCENDING

# Getting the database url
MONGO_URI = os.environ.get('MONGO_DATABASE_URL', 'mongodb://localhost:27017/nawwar') 

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

def main():
    """
    Adding tags to new articles
    """

    # Getting last entry date
    with open('last_entry.txt', 'r') as f:
        LAST_ENTRY = f.readline()

    # Setting LAST_ENTRY to a value in case of NONE
    if not LAST_ENTRY :
        LAST_ENTRY = '1990-04-29 19:17:47.829000\n'

    LAST_ENTRY = datetime.strptime(LAST_ENTRY, '%Y-%m-%d %H:%M:%S.%f\n')

    print(LAST_ENTRY)

    # connecting to mongodb
    Client = MongoClient(MONGO_URI)
    Client = Client.nawwar
    print(Client.contents.count())
    # retrieving new articles
    contents = Client["contents"].find({
        "touchDate": {"$gt": LAST_ENTRY }
        }).sort("touchDate", ASCENDING)

    for content in contents :
        # TODO: call ml_function
        function_return = model.getTags(cleanhtml(content['body']))

        function_return = list(function_return)

        function_return = [tag for tag in function_return if tag not in content['tags']]
        # adding tags
        Client.contents.find_one_and_update(
            { "_id": content['_id'] },
            { "$push" : 
                {
                    "tags": 
                    {
                        "$each": function_return
                    }
                }
            }
        )

        # updating last inserted article date
        if content['touchDate'] > LAST_ENTRY :
            LAST_ENTRY = content['touchDate']

    # updating in the last_entry.txt file
    with open('last_entry.txt', 'w') as f:
        f.write('%s\n'% LAST_ENTRY)




if __name__ == '__main__' :
    main()
