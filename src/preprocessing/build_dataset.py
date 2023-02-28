import os
from handlers import mongodbHandler, sqlHandler

def build_dataset():

    mgdb = mongodbHandler(
        url=os.environ['MONGO_URL'],
        user=os.environ['MONGO_USER'],
        password=os.environ['MONGO_PASS'],
        authMechanism=os.environ['MONGO_AUTH_MECH'],
        authSource=os.environ['MONGO_AUTH_SRC']
    )