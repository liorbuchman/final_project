import zeep
from zeep.wsse.username import UsernameToken
import config

def get_camera_limits():
    wsdl = 'http://www.onvif.org/ver20/ptz/wsdl/ptz.wsdl'
    client = zeep.Client(wsdl=wsdl, transport=zeep.transports.Transport(cache=None))
    
    ptz_service = client.create_service('{http://www.onvif.org/ver20/ptz/wsdl}PTZBinding', config.PTZ_URL)
    
    try:
        nodes = ptz_service.GetNodes()
        node = nodes[0]
        
        print("--- Camera Physical Axis Limits ---")
        pan_range = node.SupportedPTZSpaces.PanTiltSpace[0].PanTiltLimits.Range.XRange
        tilt_range = node.SupportedPTZSpaces.PanTiltSpace[0].PanTiltLimits.Range.YRange
        
        print(f"Pan Range (Azimuth): {pan_range.Min} to {pan_range.Max}")
        print(f"Tilt Range (Elevation): {tilt_range.Min} to {tilt_range.Max}")
        
    except Exception as e:
        print(f"Error fetching limits: {e}")
        print("Note: Some cameras require specific ProfileTokens.")

if __name__ == "__main__":
    get_camera_limits()