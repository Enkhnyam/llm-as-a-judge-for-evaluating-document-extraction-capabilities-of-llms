import requests, json

dois = [
  "10.1002/app.38706",
  "10.1002/cssc.201701798",
  "10.1007/s10562-021-03716-3",
  "10.1007/s10973-020-10331-8",
  "10.1016/j.catcom.2010.02.011",
  "10.1016/j.cattod.2025.115187",
  "10.1016/j.eurpolymj.2009.01.025",
  "10.1016/j.eurpolymj.2021.110590",
  "10.1016/j.polymdegradstab.2010.12.020",
  "10.1016/j.polymdegradstab.2014.10.005",
  "10.1016/j.polymdegradstab.2021.109601",
  "10.1016/j.polymdegradstab.2021.109691",
  "10.1016/j.polymdegradstab.2021.109751",
  "10.1016/j.polymdegradstab.2022.110050",
  "10.1016/j.wasman.2023.09.016",
  "10.1021/acs.iecr.5b03857",
  "10.1021/acsengineeringau.1c00039",
  "10.1021/acssuschemeng.1c04060",
  "10.1021/ie503677w",
  "10.1021/sc5007522",
  "10.1039/c2gc35696a",
  "10.1039/d0gc00327a",
  "10.1351/PAC-CON-11-06-10",
  "10.3390/polym5041258"
]
email = "enkhnyam.battulga@rwth-aachen.de"
for doi in dois:
    r = requests.get(f"https://api.unpaywall.org/v2/{doi}", params={"email": email}).json()
    loc = r.get("best_oa_location") or {}
    print(f"{doi}\t{r.get('is_oa')}\t{loc.get('host_type','')}\t{loc.get('license','')}\t{loc.get('url_for_pdf','')}")