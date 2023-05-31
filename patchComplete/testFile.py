

class ReadOFF():
    def readFile(self):
        with open(self, 'r') as file:
            if 'OFF' != file.readline().strip():
                raise ('Not a valid OFF header')
            n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
            verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
            faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
            return (verts, faces)
