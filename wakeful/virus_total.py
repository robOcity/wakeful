import requests
from .api_registration import ApiRegistration


class VirusTotal(ApiRegistration):
    def __init__(self,
                 api_key_env_var,
                 base_url='https://www.virustotal.com/vtapi/v2'):
        super().__init__(api_key_env_var, base_url)

    def get_base_url(self):
        return self.base_url

    def get_url_reputation(self, url):
        params = {'resource': url, 'apikey': self.api_key}
        vt_url = self.get_base_url() + '/url/report'
        print(vt_url)
        response = requests.post(vt_url, data=params)
        response.raise_for_status()
        return response

    def get_ip_reputation(self, ip):
        querystring = {'ip': ip, 'apikey': self.api_key}
        vt_url = self.get_base_url() + "/ip-address/report"
        response = requests.request("GET", vt_url, params=querystring)
        response.raise_for_status()
        return response
