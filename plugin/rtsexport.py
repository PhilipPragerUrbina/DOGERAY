
bl_info = {
    "name": "RTS export",
    "blender": (2, 80, 0),
    "category": "Export",
}


import bpy
import os

from bpy_extras.io_utils import ExportHelper
from bpy.props import StringProperty, BoolProperty, EnumProperty, IntProperty,FloatProperty
from bpy.types import Operator


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
    fover: IntProperty(
        name="FOV",
        description="Field of View",
        default=45
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
        file = open(self.filepath, 'w')                     
        #print(self.use_setting)
        mesh = bpy.context.object.data
        cam = bpy.context.scene.camera
     
        file.write('/exported from blender')
        file.write('\n')
        file.write( '*,%f,%f,%f,0.01,0,0,0,3,%f,%f,%f,%f' % (cam.location[0], cam.location[1], cam.location[2],self.fover, self.maxbounces,self.samples,self.backgroundi) )
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
            iqr =  principled.inputs['IOR'].default_value;
            mat = 0;
            
            if(met > 0.5):
                mat = 3;
            if(trans > 0.5):
                mat = 4;
                rough = iqr;
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
            
            file.write( '%f,%f,%f,2,%f,%f,%f,%f,0,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%s' 
            % (v1.co.x,v1.co.y,v1.co.z,col[0]*mult,col[1]*mult,col[2]*mult,rough,v2.co.x,v2.co.y,v2.co.z,mat,v3.co.x,v3.co.y,v3.co.z, face.normal[0],face.normal[1],face.normal[2],v1.normal[0],v1.normal[1],v1.normal[2],v2.normal[0],v2.normal[1],v2.normal[2],v3.normal[0],v3.normal[1],v3.normal[2],us[0], uvs[0],us[1], uvs[1],us[2], uvs[2],smooth,tex,texer) )
            file.write('\n')
       
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


