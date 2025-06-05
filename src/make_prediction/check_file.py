import eccodes # Or import eccodes as eccodes (if that's how you import it)
import sys
import traceback

# Replace with the actual path to your problematic GRIB file
# Especially the one for Dec 2014 / Jan 2015
grib_file_path = 'data/raw/grided/era5land_vars/2m_dewpoint_temperature/2015/1/2m_dewpoint_temperature_2015_01_hourly/data.grib'
# Or the 10m_u_component_of_wind file if that was the one failing in the script load

message_count = 0
error_occurred = False

print(f"Attempting to read messages from: {grib_file_path}")

try:
    # Open the GRIB file
    with open(grib_file_path, 'rb') as f:
        # Loop through all messages using codes_grib_new_from_file
        # This will return None when the end of the file is reached
        while True:
            gid = eccodes.codes_grib_new_from_file(f)
            if gid is None:
                break # End of file

            message_count += 1
            if message_count % 100 == 0: # Print progress every 100 messages
                print(f"Read message {message_count}...")

            # You could add checks here, e.g., get some keys
            # try:
            #     param_id = eccodes.codes_get(gid, 'paramId')
            #     # print(f"Message {message_count}: paramId={param_id}")
            # except eccodes.KeyValueNotFoundError:
            #     print(f"Warning: paramId not found in message {message_count}")

            # Release the message handle
            eccodes.codes_release(gid)

except eccodes.EcCodesError as e:
    print(f"\n!!! eccodes Error occurred after reading {message_count} messages:", file=sys.stderr)
    print(f"Error code: {e.code}, Message: {e.msg}", file=sys.stderr)
    traceback.print_exc()
    error_occurred = True
except Exception as e:
    print(f"\n!!! Non-eccodes Error occurred after reading {message_count} messages:", file=sys.stderr)
    traceback.print_exc()
    error_occurred = True

if not error_occurred:
    print(f"\nSuccessfully finished reading {message_count} messages.")
else:
    print(f"\nFinished with errors after reading {message_count} messages.")