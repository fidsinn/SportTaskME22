import os
from xml.etree import ElementTree as ET

def build_xml(s_dir, v_dir):
    root = os.path.join(s_dir, v_dir)
    xml_string = ''
    xml_string += '<videos>\n'
    for path, subdirs, files in os.walk(root):
        for file in files:
            #print(os.path.basename(os.path.normpath(path)), ' - ', file)
            xml_string += ' <video name=\"{}\" class=\"{}\" />\n'.format(file, os.path.basename(os.path.normpath(path)))
        #print(path, '-', subdirs, '-', files)
        #<video name="283321395.mp4" class="UNKNOWN" />
    xml_string += '<videos>'
    return xml_string

if __name__ == "__main__":
    start_directory = 'classificationTask'
    valid_directory = 'validation'
    
    xml_string = build_xml(s_dir=start_directory, v_dir=valid_directory)
    
    # with open("validation", "w") as f:
    #     f.write(ET.tostring(xml_string))
    with open('validation.xml', 'w') as f:
        f.write(xml_string)