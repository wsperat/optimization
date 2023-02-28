import logging
import pymssql
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError


class sqlHandler:

    def __init__(self, url, user, password):

        logging.info(f'Configuring mssql handler parameters.')
        self.url, self.user, self.password = url, user, password

        logging.info(f'Trying to connect to {url} database.')
        self.conn = self.connect()
        
    def connect(self):

        logging.info(f'Connect to {self.url} database...')
        connection = pymssql.connect(self.url, self.user, self.password)

        self.conn = connection        
        return connection

    def status(self):

        try:
            cur = self.conn.cursor()
            del cur
            return True
        
        except pymssql.InterfaceError:
            logging.warn('Exception while getting connection.')
            return False


class mongodbHandler:


    def __init__(self, url, user, password, authSource, authMechanism):

        self.url, self.user, self.password  = url, user, password
        self.authSource, self.authMechanism = authSource, authMechanism

        logging.info(f'Trying to connect to {url} database.')
        self.conn = self.connect()

    def connect(self):
        
        logging.info(f'Connect to {self.url} database...')
        client = MongoClient(
            self.url,
            username=self.user,
            password=self.password,
            authSource=self.authSource,
            authMechanism=self.authMechanism)
        
        self.conn = client  
        return client

    def status(self):

        try:
            self.conn.server_info()
            return True
        except ServerSelectionTimeoutError:
            return False
