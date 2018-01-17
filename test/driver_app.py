import json
from time import sleep
from wakeful.virus_total import VirusTotal

if __name__ == '__main__':
    vt_reg = VirusTotal('VIRUS_TOTAL_PUBLIC_API_KEY')
    print(vt_reg.get_base_url())
    print(vt_reg.api_key)
    response = vt_reg.get_url_reputation('www.google.com')
    with open('test/url_rep.json', 'w') as f_out:
        json.dump(response.json(), f_out)

    # awaiting api access to ip reputation data
    # sleep(1)
    #
    # # possible zeus c2 --> 104.193.186.24
    # response = vt_reg.get_ip_reputation('104.193.186.24')
    # with open('test/ip_rep.json') as f_out:
    #     json.dump(response.json(), f_out)
