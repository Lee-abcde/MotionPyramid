#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.Threading.Tasks;

namespace AI4Animation
{
    public class PreProcess : BatchProcessorAsync
    {
        private MotionEditor Editor = null;

        [MenuItem ("AI4Animation/Tools/Pre Process")]
        static void Init() {
            Window = EditorWindow.GetWindow(typeof(PreProcess));
            Scroll = Vector3.zero;
        }

        PreProcess()
        {
            BatchSize = 10;
        }

        public override void BatchCallback()
        {
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            Resources.UnloadUnusedAssets();
        }

        public override void DerivedStart()
        {
        }

        public override bool CanProcess()
        {
            return true;
        }

        public MotionEditor GetEditor() {
            if(Editor == null) {
                Editor = GameObjectExtensions.Find<MotionEditor>(true);
            }
            return Editor;
        }

        public override string GetID(Item item) {
            return Utility.GetAssetName(item.ID);
        }

        public override void DerivedRefresh() {
			
        }
        
        public override void DerivedInspector()
        {
            if(GetEditor() == null) {
                EditorGUILayout.LabelField("No editor available in scene.");
                return;
            }
            
            Editor = EditorGUILayout.ObjectField("Editor", Editor, typeof(MotionEditor), true) as MotionEditor;
            BatchSize = EditorGUILayout.IntField("Batch Size", BatchSize);
            
            if(Utility.GUIButton("Refresh", UltiDraw.DarkGrey, UltiDraw.White)) {
                LoadItems(GetEditor().Assets.ToArray());
            }
        }

        public override void DerivedInspector(Item item)
        {
        }

        private void ProcessAssets(MotionAsset asset)
        {
            asset.RemoveAllModules();
            
            asset.MirrorAxis = Axis.ZPositive;
            asset.Model = "Biped";
            if (asset.Source.FindBone("Head") != null)
            {
                asset.Scale = 0.01f;
                asset.Source.FindBone("Head").Alignment = new Vector3(90f, 0f, 0f);
                asset.Source.FindBone("LeftShoulder").Alignment = new Vector3(90f, 0f, 0f);
                asset.Source.FindBone("RightShoulder").Alignment = new Vector3(90f, 0f, 0f);
            }
            else if (asset.Source.FindBone("upperneckSite") != null)
            {
                asset.Source.FindBone("upperneckSite").Alignment = new Vector3(90f, 0f, 0f);
                asset.Source.FindBone("lshoulder").Alignment = new Vector3(90f, 0f, 0f);
                asset.Source.FindBone("rshoulder").Alignment = new Vector3(90f, 0f, 0f);
            }
            else if (asset.Source.FindBone("def_head") != null)
            {
                asset.Scale = 0.01f;
                asset.Source.FindBone("def_head").Alignment = new Vector3(90f, 0f, 0f);
                asset.Source.FindBone("def_leftShoulder").Alignment = new Vector3(90f, 0f, 0f);
                asset.Source.FindBone("def_rightShoulder").Alignment = new Vector3(90f, 0f, 0f);
            }
            else
            {
                throw new System.Exception("No known head/shoulder bones found in the BVH. Alignment failed.");
            }

            
            asset.Export = true;
            asset.ClearSequences();
            asset.AddSequence();

            {
                bool flag = asset.HasModule<RootModule>();
                RootModule module =  flag ? asset.GetModule<RootModule>() : asset.AddModule<RootModule>();
                module.Topology = RootModule.TOPOLOGY.Biped;
                module.SmoothRotations = true;
            }
        }
        
        public override async Task DerivedProcess(Item item)
        {
            await Task.Yield();
            ProcessAssets(MotionAsset.Retrieve(item.ID));
        }

        public override void DerivedFinish()
        {
            
        }
    }
}

#endif