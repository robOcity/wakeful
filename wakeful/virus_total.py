import requests
from .api_registration import ApiRegistration


class VirusTotal(ApiRegistration):
    def __init__(self,
                 api_key_env_var,
                 base_url= 'https://www.virustotal.com/vtapi/v2'):
        super().__init__(api_key_env_var, base_url)

    def get_base_url(self):
        return self.base_url

    def get_url_reputation(self, url):
        params = {'apikey': self.api_key, 'url':url}
        vt_url = self.get_base_url + '/url/scan'
        response = requests.post(vt_url, data=params)
        return response.json()

    def get_ip_reputation(self, ip):
        querystring = {"apikey":self.api_key,"resource":ip}
        vt_url = self.get_base_url +  "/comments/get"
        response = requests.request("GET", vt_url, params=querystring)
        return response.json()


if __name__ == '__main__':
    vt_reg = VirusTotal('VIRUS_TOTAL_PUBLIC_API_KEY')
    print(vt_reg.get_base_url())
    print(vt_reg.api_key)
    print(vt_reg.get_url_reputation('www.google.com'))
