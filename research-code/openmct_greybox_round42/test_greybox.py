#!/usr/bin/env python3
import unittest
import numpy as np
import pandas as pd
import run_qualification as m

class GreyboxTests(unittest.TestCase):
    def test_exact_zoh_constant_solution(self):
        th=np.array([2.0,3.0,4.0]); x=np.array([5.0]); p=np.array([7.0]); dt=np.array([.13])
        got=m.zoh(th,x,p,dt)[0]; eq=(3*7+4)/2; expected=eq+(5-eq)*np.exp(-2*.13)
        self.assertAlmostEqual(got,expected,places=13)

    def test_semigroup(self):
        th=np.array([1.7,.8,-2.]); x=np.array([3.]); p=np.array([4.])
        whole=m.zoh(th,x,p,np.array([.2]))
        half=m.zoh(th,m.zoh(th,x,p,np.array([.1])),p,np.array([.1]))
        np.testing.assert_allclose(whole,half,rtol=1e-14,atol=1e-14)

    def test_timing_contract_and_no_reference(self):
        d=pd.DataFrame({"MEAS":[0.,1.,2.,3.,4.],"PWM":[10.,20.,30.,40.,50.],"REF":[999.,-999.,5.,6.,7.],"DT_ms":[2.,3.,4.,5.,6.]})
        s,p,dt,y=m.rows(d,"pwm_k"); np.testing.assert_array_equal(s,[1,2,3]); np.testing.assert_array_equal(p,[20,30,40]); np.testing.assert_array_equal(y,[2,3,4]); np.testing.assert_allclose(dt,[.003,.004,.005])
        _,pl,_,_=m.rows(d,"pwm_km1"); np.testing.assert_array_equal(pl,[10,20,30])
        d2=d.copy(); d2.REF*=12345
        for timing in ("pwm_k","pwm_km1"):
            for a,b in zip(m.rows(d,timing),m.rows(d2,timing)): np.testing.assert_array_equal(a,b)

    def test_authorization_is_fail_closed_boolean(self):
        protocol=m.PROTOCOL
        self.assertEqual(protocol["network_simulation_authorization"],"only if every source, identifiability, holdout and integrity check passes")
        self.assertTrue(protocol["split"]["random_row_split_prohibited"])

if __name__=="__main__": unittest.main()
