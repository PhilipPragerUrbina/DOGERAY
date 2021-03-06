
bl_info = {
    "name": "RTS export",
    "blender": (2, 80, 0),
    "category": "Export",
}


import bpy
import os

from bpy_extras.io_utils import ExportHelper
from bpy.props import StringProperty, BoolProperty, EnumProperty, IntProperty,FloatProperty,FloatVectorProperty
from bpy.types import Operator

import array
import imbuf
from pathlib import Path
def file_exists(filepath):
   if os.path.exists("filepath"):
       return "true"
   else:
       return "false"

def writeppm(path):
   
    
    
   
    newpath = os.path.splitext(path)[0]+'.ppm'
    pathen = Path(newpath)
    if pathen.exists():
        print("no")
        return
    imb = imbuf.load(path)
    print("imbuf loaded")
    imbuf.write(imb, newpath)
    print("template loaded")
    imb.free()
    print("freed")

 

    image_file = os.path.basename(path) # this refers to an image file loaded into Blender
    print("name found")
    img = bpy.data.images[image_file]
    print("data loadd")
    
    pixels = img.pixels[:]
    print("pixels loaded")
    DISPLAY_WIDTH = img.size[0]  
    DISPLAY_HEIGHT =  img.size[1] 
    
    print("image loaded")

    ppm_header = f'P6\n{DISPLAY_WIDTH} {DISPLAY_HEIGHT}\n{255}\n'

    image = array.array('B', [0, 0, 255] * DISPLAY_WIDTH * DISPLAY_HEIGHT)
    print("output loaded")

    for y in range(0, DISPLAY_HEIGHT):
        for x in range(0, DISPLAY_WIDTH):
            ny = (DISPLAY_HEIGHT - y) -1
            
            index = 3 * (ny * DISPLAY_WIDTH + x)
            index2 = 4 * (y * DISPLAY_WIDTH + x)
            image[index] =   int(pixels[index2]*255)
            image[index + 1] =  int(pixels[index2+1]*255)
            image[index + 2] = int(pixels[index2+2]*255)
         
    print("proccesed")

 # Save the PPM image as a binary file
    with open(newpath, 'wb') as f:
        print("opened")
        f.write(bytearray(ppm_header, 'ascii'))
        print("header")
        image.tofile(f)
        f.close()





            
            
 

class ExportSomeData(Operator, ExportHelper):
    """ray traced scene file expoter"""
    bl_idname = "export.some_data"
    bl_label = "Export rts"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filename_ext = ".rts"

    filter_glob: StringProperty(
        default="*.rts",
        options={'HIDDEN'},
        maxlen=255,  # Max internal buffer length, longer would be clamped.
        )
    samples: IntProperty(
        name="Samples per Pixel",
        description="How many samples per update",
        default=1
        )
    maxbounces: IntProperty(
        name="Max bounces",
        description="max bounces per ray",
        default=10
        )
    brighy: IntProperty(
        name="Emmision Brightness",
        description="How much emmsion colors should be multiplied",
        default=1
        )
    backgroundi: FloatProperty(
        name="Background Intensity",
        description="How intense background light should be 1 is normal",
        default=1
        )
    fdist: FloatProperty(
        name="Focus Distance",
        description="Distance to focus to",
        default=3
        )
        
    apetur: FloatProperty(
        name="Apeture",
        description="DOF apeture",
        default=0.01
        )
    fover: IntProperty(
        name="FOV",
        description="Field of View",
        default=45
        )
        
    xres: IntProperty(
        name="Image Width",
        description="X Resolution",
        default=1280
        )
        
    yres: IntProperty(
        name="Image Height",
        description="Y Resolution",
        default=720
        )
    positioner: FloatVectorProperty(
        name = "Look At",
        description="where camera should face",
        default=(0.0, 0.0, 0.0), 
        min= 0.0,
        max = 100,
        subtype = 'XYZ'
        # 'COLOR', 'TRANSLATION', 'DIRECTION', 'VELOCITY', 
        # 'ACCELERATION', 'MATRIX', 'EULER', 'QUATERNION', 
        # 'AXISANGLE', 'XYZ', 'COLOR_GAMMA', 'LAYER'
        )
        
        
    
 
#    use_setting: BoolProperty(
#        name="Example Boolean",
#        description="Example Tooltip",
#        default=True,
#    )


#    type: EnumProperty(
#        name="Example Enum",
#        description="Choose between two items",
#        items=(
#            ('OPT_A', "First Option", "Description one"),
#            ('OPT_B', "Second Option", "Description two"),
#        ),
#        default='OPT_A',
#    )
    @classmethod
    def poll(cls, context):
        return context.object is not None

    def execute(self, context):
        bpy.ops.object.transform_apply(location = True, scale = True, rotation = True)
        paths = []
        file = open(self.filepath, 'w')   
        
        #print(self.use_setting)
        mesh = bpy.context.object.data
        cam = bpy.context.scene.camera
        
        
        C = bpy.context
        try:
            env_img = C.scene.world.node_tree.nodes['Environment Texture'].image.name
            backtex = os.path.splitext(env_img)[0]+'.ppm'
            patherr = bpy.path.abspath(C.scene.world.node_tree.nodes['Environment Texture'].image.filepath)
            if patherr not in paths:
                paths.append(patherr)
        except:
             backtex = 'no'
        
        file.write('/exported from blender')
        file.write('\n')
        file.write( '*,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%s,%i,%i' % (cam.location[0], -cam.location[2], cam.location[1],self.apetur,self.positioner[0],-self.positioner[2],self.positioner[1],self.fdist,self.fover, self.maxbounces,self.samples,self.backgroundi, backtex,self.xres,self.yres) )
        file.write('\n')
        for face in mesh.polygons:
            
            slot = bpy.context.object.material_slots[face.material_index]
            mat = slot.material
            print(mat.name)
            print( mat.diffuse_color)
          
            # Get the nodes in the node tree
            nodes = mat.node_tree.nodes
            # Get a principled node
            principled = next(n for n in nodes if n.type == 'BSDF_PRINCIPLED')
            # Get the slot for 'base color'
            base_color = principled.inputs['Base Color'] #Or principled.inputs[0]
            # Get its default value (not the value from a possible link)
            try:
                link = base_color.links[0]
                link_node = link.from_node
                print( link_node.image.name )
                
                
                
                texer = os.path.splitext(link_node.image.name)[0]+'.ppm'
                pather = bpy.path.abspath(link_node.image.filepath)
                if pather not in paths:
                    paths.append(pather)
                print(texer)
            except:
                texer = "no"
                print( 'not found' )
            col = base_color.default_value
            col[0];
            col[1];
            col[2];
            met = principled.inputs['Metallic'].default_value;
            trans = principled.inputs['Transmission'].default_value;
            em = principled.inputs['Emission'].default_value;
            rough = principled.inputs['Roughness'].default_value;
            
            try:
                linkk = principled.inputs['Roughness'].links[0]
                link_nodee = linkk.from_node
             
                
                
                
                texerr = os.path.splitext(link_nodee.image.name)[0]+'.ppm'
                patherrr = bpy.path.abspath(link_nodee.image.filepath)
                if patherrr not in paths:
                    paths.append(patherrr)
                
            except:
                texerr = "no"
               
            iqr =  principled.inputs['IOR'].default_value;
            spec =  principled.inputs['Specular'].default_value;
            mat = 0;
            
            if(met > 0.5):
                mat = 3;
            if(trans > 0.5):
                mat = 4;
                rough = iqr;
            if(spec > 0.7):
                mat = 5;
               
            mult = 1;
            if(em[0] > 0.5 or em[1] > 0.5 or em[2] > 0.5):
                mat = 1;
                mult = self.brighy;

            vs = [0,0,0,0,0,0,0]
            i = 0;
            for vert in face.vertices:
                vs[i] = vert;
                i = i+1;
                
            v1 = mesh.vertices[vs[0]]
            v2 = mesh.vertices[vs[1]]
            v3 = mesh.vertices[vs[2]]
            
            
            us = [0,0,0,0,0,0,0]
            uvs = [0,0,0,0,0,0,0]
            m = 0;
            
            for vert_idx, loop_idx in zip(face.vertices, face.loop_indices):
                uv_coords = mesh.uv_layers.active.data[loop_idx].uv
                us[m] = uv_coords.x
                uvs[m] = uv_coords.y
                m = m+1
          
            smooth = 0;
            if(face.use_smooth == True):
                smooth = 1;
            tex = 0;
            if(principled.inputs['Alpha'].default_value < 0.5):
                tex = 1;
          
           
        
      
          
                
            file.write( '%f,%f,%f,2,%f,%f,%f,%f,0,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%s,%s' 
                % (v1.co.x,-v1.co.z,v1.co.y,col[0]*mult,col[1]*mult,col[2]*mult,rough,v2.co.x,-v2.co.z,v2.co.y,mat,v3.co.x,-v3.co.z,v3.co.y, 
                face.normal[0],-face.normal[2],face.normal[1],v1.normal[0],-v1.normal[2],v1.normal[1],v2.normal[0],-v2.normal[2],v2.normal[1],v3.normal[0],-v3.normal[2],v3.normal[1],us[0], uvs[0],us[1], uvs[1],us[2], uvs[2],smooth,tex,texer,texerr) )
            file.write('\n')
        file.close()
      
       
        print("proccesing")
        print( paths)
        print("yes")
        for pathings in paths:
            print(pathings)
            writeppm(pathings)
            print("done")
        
      
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


# Only needed if you want to add into a dynamic menu
def menu_func(self, context):
    self.layout.operator_context = 'INVOKE_DEFAULT'
    self.layout.operator(ExportSomeData.bl_idname, text="RTS export")

# Register and add to the file selector
def register():
    bpy.utils.register_class(ExportSomeData)
    bpy.types.TOPBAR_MT_file_export.append(menu_func)

def unregister():
    bpy.utils.unregister_class(ExportSomeData)


