import jinja2

POPUP_TEMPLATE = jinja2.Template(
    """
<img src="{{data['country_flag']}}"> </img>
</br></br>
                                 
<b>City:</b> {{data['city']}}, {{data['country_name']}}</br>
<b>IP:</b> {{data['ip']}}</br>
<b>Continent:</b> {{data['continent_name']}}</br>

<b>Longitude:</b> {{data['longitude']}} </br>
<b>Latitude:</b> {{data['latitude']}} </br>
                         
"""
)