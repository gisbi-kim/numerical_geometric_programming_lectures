# test_oplus.py

import unittest
import numpy as np
from pose_module import oplus, exp_map, log_map, skew, unskew


class TestOplusFunction(unittest.TestCase):
    def setUp(self):
        """
        초기 설정을 수행합니다.
        주로 기본 SE(3) 행렬(항등 행렬)과 SO(3) 행렬을 설정합니다.
        """
        self.identity_SE3 = np.eye(4)
        self.identity_SO3 = np.eye(3)

    def assertSE3AlmostEqual(self, T1, T2, atol=1e-6):
        """
        두 SE(3) 행렬이 거의 동일한지 확인하는 헬퍼 함수입니다.
        """
        np.testing.assert_allclose(T1, T2, atol=atol)

    def assertSO3AlmostEqual(self, R1, R2, atol=1e-6):
        """
        두 SO(3) 행렬이 거의 동일한지 확인하는 헬퍼 함수입니다.
        """
        np.testing.assert_allclose(R1, R2, atol=atol)

    # 기존 SE(3) 관련 테스트 케이스 유지
    def test_oplus_zero_update_SE3(self):
        """
        제로 업데이트: 항등 SE(3) 행렬에 제로 벡터를 적용했을 때 항등 행렬이 유지되는지 확인합니다.
        """
        tangent_vec = np.zeros(6)
        T_updated = oplus(self.identity_SE3.copy(), tangent_vec)
        self.assertSE3AlmostEqual(T_updated, self.identity_SE3)

    def test_oplus_pure_translation_SE3(self):
        """
        순수 평행 이동 업데이트: 회전 없이 평행 이동만 적용했을 때 올바르게 업데이트되는지 확인합니다.
        """
        translation = np.array([0.5, -0.3, 1.2])
        tangent_vec = np.hstack((np.zeros(3), translation))
        T_expected = self.identity_SE3.copy()
        T_expected[:3, 3] += translation
        T_updated = oplus(self.identity_SE3.copy(), tangent_vec)
        self.assertSE3AlmostEqual(T_updated, T_expected)

    def test_oplus_pure_rotation_SE3(self):
        """
        순수 회전 업데이트: 평행 이동 없이 회전만 적용했을 때 올바르게 업데이트되는지 확인합니다.
        """
        # 90도 회전을 x축을 중심으로
        angle = np.pi / 2
        omega = np.array([angle, 0, 0])
        tangent_vec = np.hstack((omega, np.zeros(3)))
        T_updated = oplus(self.identity_SE3.copy(), tangent_vec)

        # 기대되는 회전 행렬
        R_expected = exp_map(omega)
        T_expected = np.eye(4)
        T_expected[:3, :3] = R_expected
        self.assertSE3AlmostEqual(T_updated, T_expected)

    def test_oplus_rotation_and_translation_SE3(self):
        """
        회전과 평행 이동을 동시에 업데이트했을 때 올바르게 업데이트되는지 확인합니다.
        """
        angle = np.pi / 4  # 45도 회전
        omega = np.array([0, angle, 0])  # y축을 중심으로 회전
        translation = np.array([1.0, 2.0, 3.0])
        tangent_vec = np.hstack((omega, translation))

        T_updated = oplus(self.identity_SE3.copy(), tangent_vec)

        # 기대되는 회전 행렬과 평행 이동
        R_expected = exp_map(omega)
        T_expected = np.eye(4)
        T_expected[:3, :3] = R_expected
        T_expected[:3, 3] = translation
        self.assertSE3AlmostEqual(T_updated, T_expected)

    def test_oplus_sequential_updates_SE3(self):
        """
        여러 번의 업데이트를 연속으로 적용했을 때 올바르게 누적되는지 확인합니다.
        """
        T = self.identity_SE3.copy()

        # 첫 번째 업데이트: 평행 이동
        tangent_vec1 = np.hstack((np.zeros(3), np.array([1, 0, 0])))
        T = oplus(T, tangent_vec1)

        # 두 번째 업데이트: 회전 90도 z축을 중심으로
        angle = np.pi / 2
        omega2 = np.array([0, 0, angle])
        tangent_vec2 = np.hstack((omega2, np.zeros(3)))
        T = oplus(T, tangent_vec2)

        # 세 번째 업데이트: 평행 이동
        tangent_vec3 = np.hstack((np.zeros(3), np.array([0, 1, 0])))
        T = oplus(T, tangent_vec3)

        # 기대되는 최종 SE(3) 행렬
        R_expected = exp_map(omega2)
        t_expected = np.array([1, 1, 0])
        T_expected = np.eye(4)
        T_expected[:3, :3] = R_expected
        T_expected[:3, 3] = t_expected
        self.assertSE3AlmostEqual(T, T_expected)

    def test_oplus_rotation_normalization_SE3(self):
        """
        업데이트 후 SE(3)의 회전 행렬이 항상 직교 행렬이고 행렬식이 1인지 확인합니다.
        """
        # 임의의 회전 벡터
        omega = np.array([0.3, -0.2, 0.5])
        tangent_vec = np.hstack((omega, np.zeros(3)))
        T_updated = oplus(self.identity_SE3.copy(), tangent_vec)

        R = T_updated[:3, :3]

        # 직교성 검사: R * R.T = I
        orthogonality = np.dot(R, R.T)
        np.testing.assert_allclose(orthogonality, np.eye(3), atol=1e-6)

        # 행렬식이 1인지 검사
        det = np.linalg.det(R)
        self.assertAlmostEqual(det, 1.0, places=6)

    def test_oplus_small_angle_update_SE3(self):
        """
        작은 각도의 회전 업데이트가 정확하게 처리되는지 확인합니다.
        """
        small_angle = 1e-6
        omega = np.array([small_angle, small_angle, small_angle])
        tangent_vec = np.hstack((omega, np.zeros(3)))
        T_updated = oplus(self.identity_SE3.copy(), tangent_vec)

        # 기대되는 회전 행렬 (항등 행렬에 작은 회전)
        R_expected = exp_map(omega)
        T_expected = np.eye(4)
        T_expected[:3, :3] = R_expected
        self.assertSE3AlmostEqual(T_updated, T_expected)

    def test_oplus_large_translation_SE3(self):
        """
        큰 평행 이동 업데이트가 정확하게 처리되는지 확인합니다.
        """
        large_translation = np.array([100.0, -200.0, 300.0])
        tangent_vec = np.hstack((np.zeros(3), large_translation))
        T_updated = oplus(self.identity_SE3.copy(), tangent_vec)

        T_expected = self.identity_SE3.copy()
        T_expected[:3, 3] = large_translation
        self.assertSE3AlmostEqual(T_updated, T_expected)

    # 기존 예외 테스트 수정
    def test_oplus_invalid_input_shape_SE3(self):
        """
        잘못된 형태의 SE(3) 입력 벡터에 대해 예외가 발생하는지 확인합니다.
        """
        # tangent_vec의 길이가 6이 아닌 경우
        tangent_vec = np.array([1, 2, 3])  # 길이 3
        with self.assertRaises(ValueError):
            oplus(self.identity_SE3.copy(), tangent_vec)

    def test_oplus_invalid_SE3_matrix(self):
        """
        유효하지 않은 SE(3) 행렬에 대해 함수가 어떻게 동작하는지 확인합니다.
        (예: 회전 부분이 직교 행렬이 아닌 경우)
        """
        # 회전 부분이 유효하지 않은 행렬
        invalid_SE3 = self.identity_SE3.copy()
        invalid_SE3[:3, :3] = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
        tangent_vec = np.hstack((np.zeros(3), np.array([1, 1, 1])))

        # 함수는 회전 정규화를 수행하므로 예외는 발생하지 않지만, 정규화된 회전 행렬을 반환하는지 확인
        T_updated = oplus(invalid_SE3, tangent_vec)

        R = T_updated[:3, :3]

        # 직교성 검사
        orthogonality = np.dot(R, R.T)
        np.testing.assert_allclose(orthogonality, np.eye(3), atol=1e-6)

        # 행렬식이 1인지 검사
        det = np.linalg.det(R)
        self.assertAlmostEqual(det, 1.0, places=6)

        # 평행 이동이 올바르게 업데이트되었는지 확인
        expected_translation = invalid_SE3[:3, 3] + tangent_vec[3:]
        np.testing.assert_allclose(T_updated[:3, 3], expected_translation, atol=1e-6)

    # 새로운 SO(3) 관련 테스트 케이스 추가
    def test_oplus_zero_update_SO3(self):
        """
        제로 업데이트: 항등 SO(3) 행렬에 제로 벡터를 적용했을 때 항등 행렬이 유지되는지 확인합니다.
        """
        tangent_vec = np.zeros(3)
        R_updated = oplus(self.identity_SO3.copy(), tangent_vec)
        self.assertSO3AlmostEqual(R_updated, self.identity_SO3)

    def test_oplus_pure_rotation_SO3(self):
        """
        순수 회전 업데이트: 회전만 적용했을 때 올바르게 업데이트되는지 확인합니다.
        """
        # 90도 회전을 z축을 중심으로
        angle = np.pi / 2
        omega = np.array([0, 0, angle])
        tangent_vec = omega
        R_updated = oplus(self.identity_SO3.copy(), tangent_vec)

        # 기대되는 회전 행렬
        R_expected = exp_map(omega)
        self.assertSO3AlmostEqual(R_updated, R_expected)

    def test_oplus_rotation_and_sequential_updates_SO3(self):
        """
        여러 번의 회전 업데이트를 연속으로 적용했을 때 올바르게 누적되는지 확인합니다.
        """
        R = self.identity_SO3.copy()

        # 첫 번째 업데이트: 45도 회전 x축
        angle1 = np.pi / 4
        omega1 = np.array([angle1, 0, 0])
        tangent_vec1 = omega1
        R = oplus(R, tangent_vec1)

        # 두 번째 업데이트: 45도 회전 y축
        angle2 = np.pi / 4
        omega2 = np.array([0, angle2, 0])
        tangent_vec2 = omega2
        R = oplus(R, tangent_vec2)

        # 기대되는 최종 회전 행렬
        R_expected = exp_map(omega2) @ exp_map(omega1)
        self.assertSO3AlmostEqual(R, R_expected)

    def test_oplus_rotation_normalization_SO3(self):
        """
        업데이트 후 SO(3) 회전 행렬이 항상 직교 행렬이고 행렬식이 1인지 확인합니다.
        """
        # 임의의 회전 벡터
        omega = np.array([0.3, -0.2, 0.5])
        tangent_vec = omega
        R_updated = oplus(self.identity_SO3.copy(), tangent_vec)

        # 직교성 검사: R * R.T = I
        orthogonality = np.dot(R_updated, R_updated.T)
        np.testing.assert_allclose(orthogonality, np.eye(3), atol=1e-6)

        # 행렬식이 1인지 검사
        det = np.linalg.det(R_updated)
        self.assertAlmostEqual(det, 1.0, places=6)

    def test_oplus_small_angle_update_SO3(self):
        """
        작은 각도의 회전 업데이트가 정확하게 처리되는지 확인합니다.
        """
        small_angle = 1e-6
        omega = np.array([small_angle, small_angle, small_angle])
        tangent_vec = omega
        R_updated = oplus(self.identity_SO3.copy(), tangent_vec)

        # 기대되는 회전 행렬 (항등 행렬에 작은 회전)
        R_expected = exp_map(omega)
        self.assertSO3AlmostEqual(R_updated, R_expected)

    def test_oplus_invalid_input_shape_SO3(self):
        """
        잘못된 형태의 SO(3) 입력 벡터에 대해 예외가 발생하는지 확인합니다.
        """
        # tangent_vec의 길이가 3이 아닌 경우
        tangent_vec = np.array([1, 2, 3, 4])  # 길이 4
        with self.assertRaises(ValueError):
            oplus(self.identity_SO3.copy(), tangent_vec)

    def test_oplus_invalid_SO3_matrix(self):
        """
        유효하지 않은 SO(3) 행렬에 대해 함수가 어떻게 동작하는지 확인합니다.
        (예: 회전 부분이 직교 행렬이 아닌 경우)
        """
        # 회전 부분이 유효하지 않은 행렬
        invalid_SO3 = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
        tangent_vec = np.array([1, 1, 1])

        # 함수는 회전 정규화를 수행하므로 예외는 발생하지 않지만, 정규화된 회전 행렬을 반환하는지 확인
        R_updated = oplus(invalid_SO3, tangent_vec)

        # 직교성 검사
        orthogonality = np.dot(R_updated, R_updated.T)
        np.testing.assert_allclose(orthogonality, np.eye(3), atol=1e-6)

        # 행렬식이 1인지 검사
        det = np.linalg.det(R_updated)
        self.assertAlmostEqual(det, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
