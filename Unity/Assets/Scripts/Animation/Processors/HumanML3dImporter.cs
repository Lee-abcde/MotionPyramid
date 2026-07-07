#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using NumSharp;

namespace AI4Animation {
    public class HumanML3DImporter : BatchProcessor {

        public string Source = string.Empty;
        public string Destination = string.Empty;

        private List<string> Imported;
        private List<string> Skipped;

        [MenuItem("AI4Animation/Importer/HumanML3D Importer")]
        static void Init() {
            Window = EditorWindow.GetWindow(typeof(HumanML3DImporter));
            Scroll = Vector3.zero;
            
            HumanML3DImporter importer = Window as HumanML3DImporter;
            if (importer != null) {
                importer.Source = @"D:\Learning\MyProject\ETHCourse\2024HSProj\SP\HumanML3D\HumanML3D\new_joint_posrot";
                importer.Destination = "HumanML3D";
            }
        }

        public override string GetID(Item item) {
            return item.ID;
        }

        public override void DerivedRefresh() {
        }

        public override void DerivedInspector() {
            EditorGUILayout.LabelField("Source");
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField("<Path>", GUILayout.Width(50));
            Source = EditorGUILayout.TextField(Source);
            GUI.skin.button.alignment = TextAnchor.MiddleCenter;
            if (GUILayout.Button("O", GUILayout.Width(20))) {
                Source = EditorUtility.OpenFolderPanel("HumanML3D Importer", Source == string.Empty ? Application.dataPath : Source, "");
                GUIUtility.ExitGUI();
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.LabelField("Destination");
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField("Assets/", GUILayout.Width(50));
            Destination = EditorGUILayout.TextField(Destination);
            EditorGUILayout.EndHorizontal();

            if (Utility.GUIButton("Load Source Directory", UltiDraw.DarkGrey, UltiDraw.White)) {
                LoadDirectory(Source);
            }
        }

        public override void DerivedInspector(Item item) {
        }

        private void LoadDirectory(string directory) {
            if (directory == null) {
                LoadItems(new string[0]);
            } else {
                if (Directory.Exists(directory)) {
                    List<string> paths = new List<string>();
                    Iterate(directory);
                    LoadItems(paths.ToArray());

                    void Iterate(string folder) {
                        DirectoryInfo info = new DirectoryInfo(folder);
                        // 这里我假设 HumanML3D 数据是 npy/pkl 格式
                        foreach (FileInfo i in info.GetFiles("*.pkl")) {
                            paths.Add(i.FullName);
                        }
                        foreach (FileInfo i in info.GetFiles("*.npy")) {
                            paths.Add(i.FullName);
                        }
                        foreach (DirectoryInfo i in info.GetDirectories()) {
                            Iterate(i.FullName);
                        }
                    }
                } else {
                    LoadItems(new string[0]);
                }
            }
        }

        public override bool CanProcess() {
            return true;
        }

        public override void DerivedStart() {
            Imported = new List<string>();
            Skipped = new List<string>();
        }

        public override IEnumerator DerivedProcess(Item item) {
            string source = Source;
            string destination = "Assets/" + Destination;
            string target = (destination + item.ID.Remove(0, source.Length)).Replace(".npy", "").Replace(".pkl", "");

            if(!Directory.Exists(target)) {
                Directory.CreateDirectory(target);

                FileInfo file = new FileInfo(item.ID);

                MotionAsset asset = ScriptableObject.CreateInstance<MotionAsset>();
                asset.name = file.Name;
                AssetDatabase.CreateAsset(asset, target+"/"+asset.name+".asset");
                
                NDArray npyData = np.load(file.FullName);
                int frames = npyData.shape[0];
                int Framerate = 20; // 默认帧率
                asset.Framerate = 20;
                ArrayExtensions.Resize(ref asset.Frames, frames);
                
                asset.Source = new MotionAsset.Hierarchy();
                string[] boneNames = new string[]
                {
                    "root","lhip","lknee","lankle","ltoe","ltoeSite",
                    "rhip","rknee","rankle","rtoe","rtoeSite",
                    "lowerback","upperback","chest","lowerneck","upperneck","upperneckSite",
                    "lclavicle","lshoulder","lelbow","lwrist","lwristSite",
                    "rclavicle","rshoulder","relbow","rwrist","rwristSite"
                };
                int[] parentIndices = new int[]
                {
                    -1,   // root
                    0,    // lhip -> root
                    1,    // lknee -> lhip
                    2,    // lankle -> lknee
                    3,    // ltoe -> lankle
                    4,    // ltoeSite -> ltoe
                    0,    // rhip -> root
                    6,    // rknee -> rhip
                    7,    // rankle -> rknee
                    8,    // rtoe -> rankle
                    9,    // rtoeSite -> rtoe
                    0,    // lowerback -> root
                    11,   // upperback -> lowerback
                    12,   // chest -> upperback
                    12,   // lowerneck -> chest
                    14,   // upperneck -> lowerneck
                    15,   // upperneckSite -> upperneck
                    12,   // lclavicle -> chest
                    17,   // lshoulder -> lclavicle
                    18,   // lelbow -> lshoulder
                    19,   // lwrist -> lelbow
                    20,   // lwristSite -> lwrist
                    12,   // rclavicle -> chest
                    22,   // rshoulder -> rclavicle
                    23,   // relbow -> rshoulder
                    24,   // rwrist -> relbow
                    25    // rwristSite -> rwrist
                };
                int joints = boneNames.Length;
                for(int i=0; i<joints; i++)
                {
                    string name = boneNames[i];
                    string parentName = parentIndices[i] == -1 ? "None" : boneNames[parentIndices[i]];
                    asset.Source.AddBone(name, parentName);
                }

                int[] reorderindices =
                {
                    0, 1, 4, 7, 10, 10, 2, 5, 8, 11, 11, 3, 6, 9, 12, 15, 15, 13, 16, 18, 20, 20, 14, 17, 19, 21, 21
                };
                for (int k = 0; k < frames; k++)
                {
                    Matrix4x4[] matrices = new Matrix4x4[joints];
                    for (int j = 0; j < joints; j++)
                    {
                        // possible Left-handed system we need to multiple -1 to get the correct mirror results
                        double px = npyData[k, reorderindices[j], 0] * -1;
                        double py = npyData[k, reorderindices[j], 1];
                        double pz = npyData[k, reorderindices[j], 2];
                        Vector3 pos = new Vector3((float)px, (float)py, (float)pz);

                        double qw = npyData[k, reorderindices[j], 3];
                        double qx = npyData[k, reorderindices[j], 4];
                        double qy = npyData[k, reorderindices[j], 5];
                        double qz = npyData[k, reorderindices[j], 6];
                        Quaternion rot = new Quaternion((float)qx, (float)qy, (float)qz, (float)qw);
                        rot = Quaternion.Normalize(rot);
                        matrices[j] = Matrix4x4.TRS(pos, rot, Vector3.one);
                    }

                    asset.Frames[k] = new Frame(asset, k + 1, (float)k / asset.Framerate, matrices);
                }
                
                // ===== 4. 后处理 =====
                asset.DetectSymmetry();
                asset.AddSequence();
                asset.CreateScene();

                EditorUtility.SetDirty(asset);
                Imported.Add(target);
            } else {
                Skipped.Add(target);
            }

            yield return new WaitForSeconds(0f);
        }

        public override void BatchCallback() {
            AssetDatabase.SaveAssets();
            Resources.UnloadUnusedAssets();
        }

        public override void DerivedFinish() {
            AssetDatabase.Refresh();

            Debug.Log("Imported " + Imported.Count + " assets.");
            Imported.ToArray().Print();

            Debug.Log("Skipped " + Skipped.Count + " assets.");
            Skipped.ToArray().Print();
        }
    }
}
#endif
