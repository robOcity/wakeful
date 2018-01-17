from wakeful.virus_total import VirusTotal

if __name__ == '__main__':
    vt_reg = VirusTotal('VIRUS_TOTAL_PUBLIC_API_KEY')
    print(vt_reg.get_base_url())
    print(vt_reg.api_key)
    print(vt_reg.get_url_reputation('www.google.com'))
