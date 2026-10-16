import copy,json,unittest
from pathlib import Path
from benchmarks.profiles import PROFILES,MODEL_SPECS,REPLAY_WIDTHS,TASKS,validate_profile,validate_bonsai_config,validate_hardware,validate_evaluator_version,task_protocol

class ProfileTests(unittest.TestCase):
    def test_all_twenty_builtin_profiles(self):
        self.assertEqual(len(PROFILES),20)
        self.assertEqual(set(MODEL_SPECS),{'1p7b','4b'})
        for name,p in PROFILES.items():
            self.assertEqual(validate_profile(p),p)

    def test_eight_external_profiles_match_builtins(self):
        expected={
            'bonsai_1p7b_h20_sft_ar',
            'bonsai_1p7b_h20_d4_hybrid',
            'bonsai_1p7b_h20_d8_hybrid',
            'bonsai_4b_h20_sft_ar',
            'bonsai_4b_h20_d8_hybrid',
            'bonsai_1p7b_rtx3090_d8_hybrid',
            'bonsai_1p7b_rtx3090_sft_ar',
            'bonsai_1p7b_rtx3090_d4_hybrid',
        }
        directory=Path(__file__).resolve().parents[1]/'configs/evaluation'
        self.assertEqual({path.stem for path in directory.glob('*.json')},expected)
        for name in expected:
            profile=json.loads((directory/f'{name}.json').read_text())
            self.assertEqual(validate_profile(profile),PROFILES[name])

    def test_separate_sft_baselines(self):
        for size in MODEL_SPECS:
            p=PROFILES[f'bonsai_{size}_h20_sft_ar']
            self.assertEqual(p['reference_model_id'],f'bonsai_{size}_sft')
            self.assertIsNone(p['block_size']);self.assertEqual(p['replay_widths'],[1])
            self.assertNotEqual(p['reference_model_id'],PROFILES[f'bonsai_{size}_h20_d8_ar']['reference_model_id'])

    def test_d4_schedule_and_chain(self):
        p=PROFILES['bonsai_1p7b_h20_d4_hybrid']
        self.assertEqual(p['min_chain'],3);self.assertEqual(p['replay_widths'],REPLAY_WIDTHS[4])
        self.assertEqual(p['verification'],'token_id_equality')
        self.assertNotIn('bonsai_4b_h20_d4_hybrid',PROFILES)

    def test_reject_unsupported_platform_scopes(self):
        p=copy.deepcopy(PROFILES['bonsai_1p7b_rtx3090_d8_hybrid'])
        for change in ({'model_size':'4b'},{'block_size':4},{'draft_projection':'fp8'}):
            with self.assertRaises(ValueError):validate_profile(dict(p,**change))

    def test_rtx3090_sft_is_a_separate_native_ar_checkpoint(self):
        p=PROFILES['bonsai_1p7b_rtx3090_sft_ar']
        self.assertEqual(p['checkpoint_role'],'sft')
        self.assertEqual(p['reference_model_id'],'bonsai_1p7b_sft')
        self.assertEqual(p['mode'],'ar');self.assertIsNone(p['block_size'])
        self.assertEqual(p['replay_widths'],[1]);self.assertEqual(p['draft_projection'],'native')
        self.assertNotEqual(p['reference_model_id'],PROFILES['bonsai_1p7b_rtx3090_d8_ar']['reference_model_id'])
        for changes in ({'checkpoint_role':'mtp'},{'reference_model_id':'bonsai_1p7b_mtp_d8'},
                        {'block_size':8},{'draft_projection':'bf16_raw'},{'mode':'hybrid'}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):validate_profile(dict(p,**changes))

    def test_rtx3090_d4_hybrid_contract_and_wrong_checkpoint_guards(self):
        p=PROFILES['bonsai_1p7b_rtx3090_d4_hybrid']
        self.assertEqual(p['checkpoint_role'],'mtp');self.assertEqual(p['block_size'],4)
        self.assertEqual(p['reference_model_id'],'bonsai_1p7b_mtp_d4')
        self.assertEqual(p['mode'],'hybrid');self.assertEqual(p['threshold'],0.9)
        self.assertEqual(p['min_chain'],3);self.assertEqual(p['replay_widths'],REPLAY_WIDTHS[4])
        self.assertEqual(p['draft_projection'],'bf16_raw');self.assertEqual(p['verification'],'token_id_equality')
        for changes in ({'checkpoint_role':'sft'},{'reference_model_id':'bonsai_1p7b_mtp_d8'},
                        {'block_size':8},{'min_chain':6},{'draft_projection':'fp8'},
                        {'verification':'logit_gap'},{'mode':'speculative'}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):validate_profile(dict(p,**changes))
        cfg=dict(MODEL_SPECS['1p7b'],kernel_kind='t2g128',block_size=4,causal_attn=True,mask_token_id=151654)
        validate_bonsai_config(p,cfg)
        with self.assertRaises(ValueError):validate_bonsai_config(p,dict(cfg,block_size=8))

    def test_rtx3090_profile_science_matches_h20_except_projection_and_environment(self):
        for stem,mode in (('sft','ar'),('d4','hybrid')):
            h20=PROFILES[f'bonsai_1p7b_h20_{stem}_{mode}']
            rtx=PROFILES[f'bonsai_1p7b_rtx3090_{stem}_{mode}']
            changed={key for key in h20 if h20[key]!=rtx[key]}
            expected={'device_profile','lm_eval_version'}
            if mode=='hybrid':expected.add('draft_projection')
            self.assertEqual(changed,expected)
            self.assertEqual(rtx['lm_eval_version'],'0.4.11')
            self.assertEqual(rtx['segment_ladder'],[256,1024,4096])

    def test_4b_is_h20_d8_only(self):
        names={name for name in PROFILES if name.startswith('bonsai_4b_')}
        expected={'bonsai_4b_h20_sft_ar'} | {f'bonsai_4b_h20_d8_{mode}' for mode in ('ar','confidence','speculative','hybrid')}
        self.assertEqual(names,expected)
        p=PROFILES['bonsai_4b_h20_d8_hybrid']
        self.assertEqual(p['min_chain'],6)
        self.assertEqual(p['replay_widths'],[1,8,9,10,12,16,17])
        self.assertEqual(p['verification'],'token_id_equality')
        with self.assertRaises(ValueError):validate_profile(dict(p,block_size=4))
        with self.assertRaises(ValueError):validate_profile(dict(p,device_profile='rtx3090'))

    def test_no_verifier_tolerance_field(self):
        p=PROFILES['bonsai_1p7b_h20_d8_hybrid']
        for key in ('rabv_margin','verifier_margin'):
            with self.assertRaises(ValueError):validate_profile(dict(p,**{key:0.5}))
        with self.assertRaises(ValueError):validate_profile(dict(p,verification='logit_gap'))

    def test_exact_dimensions_and_training_contract(self):
        for name,p in PROFILES.items():
            cfg=dict(MODEL_SPECS[p['model_size']],kernel_kind='t2g128',block_size=p['block_size'],causal_attn=True,mask_token_id=151654)
            validate_bonsai_config(p,cfg)
            with self.assertRaises(ValueError):validate_bonsai_config(p,dict(cfg,head_dim=64))
            if p['checkpoint_role']=='mtp':
                with self.assertRaises(ValueError):validate_bonsai_config(p,dict(cfg,block_size=2))

    def test_hardware_and_evaluator_preflight(self):
        for platform,gpu,sm,version in [('h20','NVIDIA H20',(9,0),'0.4.9.2'),('rtx3090','NVIDIA GeForce RTX 3090',(8,6),'0.4.11')]:
            p=PROFILES[f'bonsai_1p7b_{platform}_d8_hybrid']
            validate_hardware(p,device_name=gpu,compute_capability=sm,visible_count=1)
            validate_evaluator_version(p,version)
            with self.assertRaises(ValueError):validate_hardware(p,device_name=gpu,compute_capability=sm,visible_count=2)
            with self.assertRaises(ValueError):validate_evaluator_version(p,'0.0')

    def test_task_protocols(self):
        self.assertEqual(set(TASKS),{'gsm8k','humaneval_plus'})
        for platform in ('h20','rtx3090'):
            p=task_protocol('gsm8k',device_profile=platform)
            self.assertTrue(p['chat']);self.assertTrue(p['multiturn']);self.assertEqual(p['fewshot'],4)
            self.assertEqual(task_protocol('humaneval_plus',device_profile=platform)['max_gen'],1024)
        with self.assertRaises(ValueError):task_protocol('gsm8k',max_gen=12)
        for task in ('ifeval','math500','mbpp_plus'):
            with self.assertRaises(ValueError):task_protocol(task)

if __name__=='__main__':unittest.main()
