import os
import abc


class ApiRegistration(abc.ABC):
    def __init__(self,
                 api_key_env_var,
                 base_url):
        self.api_key_env_var = api_key_env_var
        self.api_key = self.lookup(self.api_key_env_var)
        self.base_url = base_url


    def lookup(self, env_var):
        """
        Search the current runtime environment for the environment
        variable.
        :param env_var: Environment variable to resolve
        :return: Value of the environment variable, or None, if it is not found
        """
        env_value = os.getenv(env_var)
        return env_value if env_value else None

    def get_base_url():
        """
        Retrieves the base URL for the RESTful API.
        """
        return self.base_url
        
