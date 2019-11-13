import os


class ConfigLoader:

    @staticmethod
    def load_database_connection_info():
        connection_info = {
            'hostname': os.environ.get('MYSQL_HOSTNAME'),
            'port': int(os.environ.get('MYSQL_PORT')),
            'username': os.environ.get('MYSQL_USERNAME'),
            'password': os.environ.get('MYSQL_PASSWORD'),
            'database': os.environ.get('MYSQL_DATABASE')
        }
        return connection_info

    @staticmethod
    def load_settings():
        settings = {
            'random_seed': int(os.environ.get('RANDOM_SEED'))
        }
        return settings
