use nalgebra::{Quaternion, UnitQuaternion, UnitVector3, Vector3};
use std::ops::Index;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Copy, Clone)]
pub struct Rotation {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}
#[wasm_bindgen]
impl Rotation {
    #[wasm_bindgen(constructor)]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }
    #[wasm_bindgen(getter)]
    pub fn default() -> Self {
        Self::new(0.0, 0.0, 0.0, 1.0)
    }
    #[wasm_bindgen(getter)]
    pub fn x(&self) -> f32 {
        self.x
    }

    #[wasm_bindgen(getter)]
    pub fn y(&self) -> f32 {
        self.y
    }

    #[wasm_bindgen(getter)]
    pub fn z(&self) -> f32 {
        self.z
    }

    #[wasm_bindgen(getter)]
    pub fn w(&self) -> f32 {
        self.w
    }

    pub(crate) fn to_unit_quaternion(&self) -> UnitQuaternion<f32> {
        UnitQuaternion::from_quaternion(Quaternion::new(self.w, self.x, self.y, self.z))
    }
}

#[wasm_bindgen]
pub struct PoseSolverResult {
    pub upper_body: Rotation,
    pub lower_body: Rotation,
    pub neck: Rotation,
    pub left_hip: Rotation,
    pub right_hip: Rotation,
    pub left_foot: Rotation,
    pub right_foot: Rotation,
    pub left_upper_arm: Rotation,
    pub right_upper_arm: Rotation,
    pub left_lower_arm: Rotation,
    pub right_lower_arm: Rotation,
    pub left_wrist: Rotation,
    pub right_wrist: Rotation,
    pub left_thumb_cmc: Rotation,
    pub left_thumb_mcp: Rotation,
    pub left_index_finger_mcp: Rotation,
    pub left_index_finger_pip: Rotation,
    pub left_index_finger_dip: Rotation,
    pub left_middle_finger_mcp: Rotation,
    pub left_middle_finger_pip: Rotation,
    pub left_middle_finger_dip: Rotation,
    pub left_ring_finger_mcp: Rotation,
    pub left_ring_finger_pip: Rotation,
    pub left_ring_finger_dip: Rotation,
    pub left_pinky_finger_mcp: Rotation,
    pub left_pinky_finger_pip: Rotation,
    pub left_pinky_finger_dip: Rotation,
    pub right_thumb_cmc: Rotation,
    pub right_thumb_mcp: Rotation,
    pub right_index_finger_mcp: Rotation,
    pub right_index_finger_pip: Rotation,
    pub right_index_finger_dip: Rotation,
    pub right_middle_finger_mcp: Rotation,
    pub right_middle_finger_pip: Rotation,
    pub right_middle_finger_dip: Rotation,
    pub right_ring_finger_mcp: Rotation,
    pub right_ring_finger_pip: Rotation,
    pub right_ring_finger_dip: Rotation,
    pub right_pinky_finger_mcp: Rotation,
    pub right_pinky_finger_pip: Rotation,
    pub right_pinky_finger_dip: Rotation,
}

impl PoseSolverResult {
    pub fn new() -> Self {
        Self {
            upper_body: Rotation::default(),
            lower_body: Rotation::default(),
            neck: Rotation::default(),
            left_hip: Rotation::default(),
            right_hip: Rotation::default(),
            left_foot: Rotation::default(),
            right_foot: Rotation::default(),
            left_upper_arm: Rotation::default(),
            right_upper_arm: Rotation::default(),
            left_lower_arm: Rotation::default(),
            right_lower_arm: Rotation::default(),
            left_wrist: Rotation::default(),
            right_wrist: Rotation::default(),
            left_thumb_cmc: Rotation::default(),
            left_thumb_mcp: Rotation::default(),
            left_index_finger_mcp: Rotation::default(),
            left_index_finger_pip: Rotation::default(),
            left_index_finger_dip: Rotation::default(),
            left_middle_finger_mcp: Rotation::default(),
            left_middle_finger_pip: Rotation::default(),
            left_middle_finger_dip: Rotation::default(),
            left_ring_finger_mcp: Rotation::default(),
            left_ring_finger_pip: Rotation::default(),
            left_ring_finger_dip: Rotation::default(),
            left_pinky_finger_mcp: Rotation::default(),
            left_pinky_finger_pip: Rotation::default(),
            left_pinky_finger_dip: Rotation::default(),
            right_thumb_cmc: Rotation::default(),
            right_thumb_mcp: Rotation::default(),
            right_index_finger_mcp: Rotation::default(),
            right_index_finger_pip: Rotation::default(),
            right_index_finger_dip: Rotation::default(),
            right_middle_finger_mcp: Rotation::default(),
            right_middle_finger_pip: Rotation::default(),
            right_middle_finger_dip: Rotation::default(),
            right_ring_finger_mcp: Rotation::default(),
            right_ring_finger_pip: Rotation::default(),
            right_ring_finger_dip: Rotation::default(),
            right_pinky_finger_mcp: Rotation::default(),
            right_pinky_finger_pip: Rotation::default(),
            right_pinky_finger_dip: Rotation::default(),
        }
    }
}

#[repr(u8)]
pub enum MainBodyIndex {
    Nose = 0,
    LeftEyeInner = 1,
    LeftEye = 2,
    LeftEyeOuter = 3,
    RightEyeInner = 4,
    RightEye = 5,
    RightEyeOuter = 6,
    LeftEar = 7,
    RightEar = 8,
    MouthLeft = 9,
    MouthRight = 10,
    LeftShoulder = 11,
    RightShoulder = 12,
    LeftElbow = 13,
    RightElbow = 14,
    LeftWrist = 15,
    RightWrist = 16,
    LeftPinky = 17,
    RightPinky = 18,
    LeftIndex = 19,
    RightIndex = 20,
    LeftThumb = 21,
    RightThumb = 22,
    LeftHip = 23,
    RightHip = 24,
    LeftKnee = 25,
    RightKnee = 26,
    LeftAnkle = 27,
    RightAnkle = 28,
    LeftHeel = 29,
    RightHeel = 30,
    LeftFootIndex = 31,
    RightFootIndex = 32,
}

pub enum HandIndex {
    Wrist = 0,
    ThumbCMC = 1,
    ThumbMCP = 2,
    ThumbIP = 3,
    ThumbTip = 4,
    IndexMCP = 5,
    IndexPIP = 6,
    IndexDIP = 7,
    IndexTip = 8,
    MiddleMCP = 9,
    MiddlePIP = 10,
    MiddleDIP = 11,
    MiddleTip = 12,
    RingMCP = 13,
    RingPIP = 14,
    RingDIP = 15,
    RingTip = 16,
    PinkyMCP = 17,
    PinkyPIP = 18,
    PinkyDIP = 19,
    PinkyTip = 20,
}

impl Index<MainBodyIndex> for Vec<Vector3<f32>> {
    type Output = Vector3<f32>;

    fn index(&self, index: MainBodyIndex) -> &Self::Output {
        &self[index as usize]
    }
}

impl Index<HandIndex> for Vec<Vector3<f32>> {
    type Output = Vector3<f32>;

    fn index(&self, index: HandIndex) -> &Self::Output {
        &self[index as usize]
    }
}

const LEFT: u8 = 0;
const RIGHT: u8 = 1;

fn landmarks_to_vector3(landmarks: js_sys::Array) -> Vec<Vector3<f32>> {
    landmarks
        .iter()
        .map(|item| {
            let obj = js_sys::Object::from(item);
            let x = js_sys::Reflect::get(&obj, &"x".into())
                .unwrap()
                .as_f64()
                .unwrap() as f32;
            let y = js_sys::Reflect::get(&obj, &"y".into())
                .unwrap()
                .as_f64()
                .unwrap() as f32;
            let z = js_sys::Reflect::get(&obj, &"z".into())
                .unwrap()
                .as_f64()
                .unwrap() as f32;
            Vector3::new(x, y, z)
        })
        .collect()
}

#[wasm_bindgen]
pub struct PoseSolver {}
#[wasm_bindgen]
impl PoseSolver {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {}
    }
    #[wasm_bindgen(js_name = solve)]
    pub fn solve(
        &mut self,
        main_body: js_sys::Array,
        left_hand: js_sys::Array,
        right_hand: js_sys::Array,
    ) -> PoseSolverResult {
        if main_body.length() == 0 {
            return PoseSolverResult::new();
        }
        let main_body: Vec<Vector3<f32>> = landmarks_to_vector3(main_body);

        let mut result = PoseSolverResult::new();

        let left_shoulder = main_body[MainBodyIndex::LeftShoulder];
        let right_shoulder = main_body[MainBodyIndex::RightShoulder];
        let left_hip = main_body[MainBodyIndex::LeftHip];
        let right_hip = main_body[MainBodyIndex::RightHip];

        result.upper_body = self.calculate_upper_body_rotation(&left_shoulder, &right_shoulder);
        result.lower_body = self.calculate_lower_body_rotation(&left_hip, &right_hip);
        result.neck = self.calculate_neck_rotation(
            &main_body[MainBodyIndex::Nose],
            &left_shoulder,
            &right_shoulder,
            &result.upper_body,
        );
        result.left_upper_arm = self.calculate_upper_arm_rotation(
            &left_shoulder,
            &main_body[MainBodyIndex::LeftElbow],
            &result.upper_body,
            LEFT,
        );
        result.left_lower_arm = self.calculate_lower_arm_rotation(
            &main_body[MainBodyIndex::LeftElbow],
            &main_body[MainBodyIndex::LeftWrist],
            &result.left_upper_arm,
            LEFT,
        );
        result.right_upper_arm = self.calculate_upper_arm_rotation(
            &right_shoulder,
            &main_body[MainBodyIndex::RightElbow],
            &result.upper_body,
            RIGHT,
        );
        result.right_lower_arm = self.calculate_lower_arm_rotation(
            &main_body[MainBodyIndex::RightElbow],
            &main_body[MainBodyIndex::RightWrist],
            &result.right_upper_arm,
            RIGHT,
        );
        result.left_hip = self.calculate_hip_rotation(
            &left_hip,
            &main_body[MainBodyIndex::LeftKnee],
            &result.lower_body,
        );
        result.right_hip = self.calculate_hip_rotation(
            &right_hip,
            &main_body[MainBodyIndex::RightKnee],
            &result.lower_body,
        );
        result.left_foot = result.left_hip;
        result.right_foot = result.right_hip;

        if left_hand.length() > 0 {
            let left_hand: Vec<Vector3<f32>> = landmarks_to_vector3(left_hand);

            result.left_wrist = self.calculate_wrist_rotation(
                &left_hand[HandIndex::Wrist],
                &left_hand[HandIndex::MiddleMCP],
                &result.left_lower_arm,
                LEFT,
            );

            result.left_thumb_cmc = self.calculate_thumb_rotation(
                &left_hand[HandIndex::ThumbCMC],
                &left_hand[HandIndex::ThumbMCP],
                &result.left_wrist,
                LEFT,
            );
            result.left_thumb_mcp = self.calculate_thumb_rotation(
                &left_hand[HandIndex::ThumbMCP],
                &left_hand[HandIndex::ThumbIP],
                &result.left_thumb_cmc,
                LEFT,
            );
            result.left_index_finger_mcp = self.calculate_finger_rotation(
                &left_hand[HandIndex::IndexMCP],
                &left_hand[HandIndex::IndexPIP],
                &result.left_wrist,
                LEFT,
            );
            result.left_index_finger_pip = self.calculate_finger_rotation(
                &left_hand[HandIndex::IndexPIP],
                &left_hand[HandIndex::IndexDIP],
                &result.left_index_finger_mcp,
                LEFT,
            );
            result.left_index_finger_dip = self.calculate_finger_rotation(
                &left_hand[HandIndex::IndexDIP],
                &left_hand[HandIndex::IndexTip],
                &result.left_index_finger_pip,
                LEFT,
            );
            result.left_middle_finger_mcp = self.calculate_finger_rotation(
                &left_hand[HandIndex::MiddleMCP],
                &left_hand[HandIndex::MiddlePIP],
                &result.left_wrist,
                LEFT,
            );
            result.left_middle_finger_pip = self.calculate_finger_rotation(
                &left_hand[HandIndex::MiddlePIP],
                &left_hand[HandIndex::MiddleDIP],
                &result.left_middle_finger_mcp,
                LEFT,
            );
            result.left_middle_finger_dip = self.calculate_finger_rotation(
                &left_hand[HandIndex::MiddleDIP],
                &left_hand[HandIndex::MiddleTip],
                &result.left_middle_finger_pip,
                LEFT,
            );
            result.left_ring_finger_mcp = self.calculate_finger_rotation(
                &left_hand[HandIndex::RingMCP],
                &left_hand[HandIndex::RingPIP],
                &result.left_wrist,
                LEFT,
            );
            result.left_ring_finger_pip = self.calculate_finger_rotation(
                &left_hand[HandIndex::RingPIP],
                &left_hand[HandIndex::RingDIP],
                &result.left_ring_finger_mcp,
                LEFT,
            );
            result.left_ring_finger_dip = self.calculate_finger_rotation(
                &left_hand[HandIndex::RingDIP],
                &left_hand[HandIndex::RingTip],
                &result.left_ring_finger_pip,
                LEFT,
            );
            result.left_pinky_finger_mcp = self.calculate_finger_rotation(
                &left_hand[HandIndex::PinkyMCP],
                &left_hand[HandIndex::PinkyPIP],
                &result.left_wrist,
                LEFT,
            );
            result.left_pinky_finger_pip = self.calculate_finger_rotation(
                &left_hand[HandIndex::PinkyPIP],
                &left_hand[HandIndex::PinkyDIP],
                &result.left_pinky_finger_mcp,
                LEFT,
            );
            result.left_pinky_finger_dip = self.calculate_finger_rotation(
                &left_hand[HandIndex::PinkyDIP],
                &left_hand[HandIndex::PinkyTip],
                &result.left_pinky_finger_pip,
                LEFT,
            );
        }

        if right_hand.length() > 0 {
            let right_hand: Vec<Vector3<f32>> = landmarks_to_vector3(right_hand);

            result.right_wrist = self.calculate_wrist_rotation(
                &right_hand[HandIndex::Wrist],
                &right_hand[HandIndex::MiddleMCP],
                &result.right_lower_arm,
                RIGHT,
            );

            result.right_thumb_cmc = self.calculate_thumb_rotation(
                &right_hand[HandIndex::ThumbCMC],
                &right_hand[HandIndex::ThumbMCP],
                &result.right_wrist,
                RIGHT,
            );
            result.right_thumb_mcp = self.calculate_thumb_rotation(
                &right_hand[HandIndex::ThumbMCP],
                &right_hand[HandIndex::ThumbIP],
                &result.right_thumb_cmc,
                RIGHT,
            );
            result.right_index_finger_mcp = self.calculate_finger_rotation(
                &right_hand[HandIndex::IndexMCP],
                &right_hand[HandIndex::IndexPIP],
                &result.right_wrist,
                RIGHT,
            );
            result.right_index_finger_pip = self.calculate_finger_rotation(
                &right_hand[HandIndex::IndexPIP],
                &right_hand[HandIndex::IndexDIP],
                &result.right_index_finger_mcp,
                RIGHT,
            );
            result.right_index_finger_dip = self.calculate_finger_rotation(
                &right_hand[HandIndex::IndexDIP],
                &right_hand[HandIndex::IndexTip],
                &result.right_index_finger_pip,
                RIGHT,
            );
            result.right_middle_finger_mcp = self.calculate_finger_rotation(
                &right_hand[HandIndex::MiddleMCP],
                &right_hand[HandIndex::MiddlePIP],
                &result.right_wrist,
                RIGHT,
            );
            result.right_middle_finger_pip = self.calculate_finger_rotation(
                &right_hand[HandIndex::MiddlePIP],
                &right_hand[HandIndex::MiddleDIP],
                &result.right_middle_finger_mcp,
                RIGHT,
            );
            result.right_middle_finger_dip = self.calculate_finger_rotation(
                &right_hand[HandIndex::MiddleDIP],
                &right_hand[HandIndex::MiddleTip],
                &result.right_middle_finger_pip,
                RIGHT,
            );
            result.right_ring_finger_mcp = self.calculate_finger_rotation(
                &right_hand[HandIndex::RingMCP],
                &right_hand[HandIndex::RingPIP],
                &result.right_wrist,
                RIGHT,
            );
            result.right_ring_finger_pip = self.calculate_finger_rotation(
                &right_hand[HandIndex::RingPIP],
                &right_hand[HandIndex::RingDIP],
                &result.right_ring_finger_mcp,
                RIGHT,
            );
            result.right_ring_finger_dip = self.calculate_finger_rotation(
                &right_hand[HandIndex::RingDIP],
                &right_hand[HandIndex::RingTip],
                &result.right_ring_finger_pip,
                RIGHT,
            );
            result.right_pinky_finger_mcp = self.calculate_finger_rotation(
                &right_hand[HandIndex::PinkyMCP],
                &right_hand[HandIndex::PinkyPIP],
                &result.right_wrist,
                RIGHT,
            );
            result.right_pinky_finger_pip = self.calculate_finger_rotation(
                &right_hand[HandIndex::PinkyPIP],
                &right_hand[HandIndex::PinkyDIP],
                &result.right_pinky_finger_mcp,
                RIGHT,
            );
            result.right_pinky_finger_dip = self.calculate_finger_rotation(
                &right_hand[HandIndex::PinkyDIP],
                &right_hand[HandIndex::PinkyTip],
                &result.right_pinky_finger_pip,
                RIGHT,
            );
        }

        result
    }

    fn calculate_upper_body_rotation(
        &self,
        left_shoulder: &Vector3<f32>,
        right_shoulder: &Vector3<f32>,
    ) -> Rotation {
        let mut spine_dir = (left_shoulder - right_shoulder).normalize();
        spine_dir.y *= -1.0;
        let default_dir = Vector3::new(1.0, 0.0, 0.0);

        let spine_rotation = UnitQuaternion::rotation_between(&default_dir, &spine_dir)
            .unwrap_or_else(UnitQuaternion::identity);

        let shoulder_center = (left_shoulder + right_shoulder) / 2.0;
        let hip_center = Vector3::zeros();
        let mut bend_dir = (shoulder_center - hip_center).normalize();
        bend_dir.y *= -1.0;

        let bend_angle = bend_dir.dot(&Vector3::y()).acos();
        let bend_axis = UnitVector3::new_normalize(Vector3::y().cross(&bend_dir));

        let bend_rotation = UnitQuaternion::from_axis_angle(&bend_axis, bend_angle);

        let quat = (spine_rotation * bend_rotation).into_inner();
        Rotation::new(quat[0], quat[1], quat[2], quat[3])
    }

    fn calculate_lower_body_rotation(
        &self,
        left_hip: &Vector3<f32>,
        right_hip: &Vector3<f32>,
    ) -> Rotation {
        let mut hip_dir = (left_hip - right_hip).normalize();
        hip_dir.y *= -1.0;
        let default_dir = Vector3::new(1.0, 0.0, 0.0);

        let quat = UnitQuaternion::rotation_between(&default_dir, &hip_dir)
            .unwrap_or_else(UnitQuaternion::identity)
            .into_inner();

        Rotation::new(quat[0], quat[1], quat[2], quat[3])
    }

    fn calculate_neck_rotation(
        &self,
        nose: &Vector3<f32>,
        left_shoulder: &Vector3<f32>,
        right_shoulder: &Vector3<f32>,
        upper_body_rotation: &Rotation,
    ) -> Rotation {
        let neck_pos = (left_shoulder + right_shoulder) / 2.0;
        let neck_dir = (nose - neck_pos).normalize();

        let upper_body_quat = upper_body_rotation.to_unit_quaternion();
        let local_neck_dir = upper_body_quat.inverse() * neck_dir;

        let forward_dir = Vector3::new(-local_neck_dir.x, 0.0, -local_neck_dir.z).normalize();
        let tilt_angle = (-local_neck_dir.y).atan2(forward_dir.magnitude());

        let tilt_offset = -std::f32::consts::PI / 9.0;
        let adjusted_tilt_angle = tilt_angle + tilt_offset;

        let horizontal_quat = UnitQuaternion::face_towards(&forward_dir, &Vector3::y());

        let tilt_quat = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), adjusted_tilt_angle);

        let quat = (horizontal_quat * tilt_quat).into_inner();

        Rotation::new(quat[0], quat[1], quat[2], quat[3])
    }

    fn calculate_upper_arm_rotation(
        &self,
        shoulder: &Vector3<f32>,
        elbow: &Vector3<f32>,
        upper_body_rotation: &Rotation,
        side: u8,
    ) -> Rotation {
        let mut arm_dir = (elbow - shoulder).normalize();
        arm_dir.y *= -1.0;

        let upper_body_quat = upper_body_rotation.to_unit_quaternion();

        let local_arm_dir = upper_body_quat.inverse() * arm_dir;

        let default_dir =
            Vector3::new(if side == LEFT { 1.0 } else { -1.0 }, -1.0, 0.0).normalize();

        let quat = UnitQuaternion::rotation_between(&default_dir, &local_arm_dir)
            .unwrap_or_else(UnitQuaternion::identity)
            .into_inner();

        Rotation::new(quat[0], quat[1], quat[2], quat[3])
    }

    fn calculate_lower_arm_rotation(
        &self,
        elbow: &Vector3<f32>,
        wrist: &Vector3<f32>,
        upper_arm_rotation: &Rotation,
        side: u8,
    ) -> Rotation {
        let mut lower_arm_dir = (wrist - elbow).normalize();
        lower_arm_dir.y *= -1.0;

        let upper_arm_quat = upper_arm_rotation.to_unit_quaternion();

        let local_lower_arm_dir = upper_arm_quat.inverse() * lower_arm_dir;

        let default_dir =
            Vector3::new(if side == LEFT { 1.0 } else { -1.0 }, -1.0, 0.0).normalize();

        let quat = UnitQuaternion::rotation_between(&default_dir, &local_lower_arm_dir)
            .unwrap_or_else(UnitQuaternion::identity)
            .into_inner();

        Rotation::new(quat[0], quat[1], quat[2], quat[3])
    }

    fn calculate_hip_rotation(
        &self,
        hip: &Vector3<f32>,
        knee: &Vector3<f32>,
        lower_body_rotation: &Rotation,
    ) -> Rotation {
        let mut leg_dir = (knee - hip).normalize();
        leg_dir.y *= -1.0;

        let lower_body_quat = lower_body_rotation.to_unit_quaternion();

        let local_leg_dir = lower_body_quat.inverse() * leg_dir;

        let default_dir = Vector3::new(0.0, -1.0, 0.0);

        let angle = default_dir.angle(&local_leg_dir);

        let max_angle = std::f32::consts::FRAC_PI_2;

        let clamped_angle = angle.min(max_angle);

        let rotation_axis = UnitVector3::new_normalize(default_dir.cross(&local_leg_dir));

        let quat = UnitQuaternion::from_axis_angle(&rotation_axis, clamped_angle).into_inner();

        Rotation::new(quat[0], quat[1], quat[2], quat[3])
    }

    fn calculate_wrist_rotation(
        &self,
        wrist: &Vector3<f32>,
        middle_finger: &Vector3<f32>,
        lower_arm_rotation: &Rotation,
        side: u8,
    ) -> Rotation {
        let mut wrist_dir = (middle_finger - wrist).normalize();
        wrist_dir.y *= -1.0;

        let lower_arm_quat = lower_arm_rotation.to_unit_quaternion();

        let local_wrist_dir = lower_arm_quat.inverse() * wrist_dir;

        let default_dir = Vector3::new(if side == LEFT { 1.0 } else { -1.0 }, -1.0, -1.0);

        let quat = UnitQuaternion::rotation_between(&default_dir, &local_wrist_dir)
            .unwrap_or_else(UnitQuaternion::identity)
            .into_inner();

        Rotation::new(quat[0], quat[1], quat[2], quat[3])
    }

    fn calculate_thumb_rotation(
        &self,
        current_joint: &Vector3<f32>,
        next_joint: &Vector3<f32>,
        parent_rotation: &Rotation,
        side: u8,
    ) -> Rotation {
        let mut joint_dir = (next_joint - current_joint).normalize();
        joint_dir.y *= -1.0;

        let parent_quat = parent_rotation.to_unit_quaternion();
        let local_joint_dir = parent_quat.inverse() * joint_dir;

        let default_dir =
            Vector3::new(if side == LEFT { -1.0 } else { 1.0 }, -1.0, -1.0).normalize();

        let quat = UnitQuaternion::rotation_between(&default_dir, &local_joint_dir)
            .unwrap_or_else(UnitQuaternion::identity)
            .into_inner();

        Rotation::new(quat[0], quat[1], quat[2], quat[3])
    }

    fn calculate_finger_rotation(
        &self,
        current_joint: &Vector3<f32>,
        next_joint: &Vector3<f32>,
        parent_rotation: &Rotation,
        side: u8,
    ) -> Rotation {
        let mut joint_dir = (next_joint - current_joint).normalize();
        joint_dir.y *= -1.0;

        let parent_quat = parent_rotation.to_unit_quaternion();

        let local_joint_dir = parent_quat.inverse() * joint_dir;

        let default_dir =
            Vector3::new(if side == LEFT { 1.0 } else { -1.0 }, -1.0, 0.0).normalize();

        let quat = UnitQuaternion::rotation_between(&default_dir, &local_joint_dir)
            .unwrap_or_else(UnitQuaternion::identity)
            .into_inner();

        Rotation::new(quat[0], quat[1], quat[2], quat[3])
    }
}
