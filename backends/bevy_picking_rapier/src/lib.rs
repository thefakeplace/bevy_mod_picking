//! A raycasting backend for `bevy_mod_picking` that uses `rapier` for raycasting.
//!
//! # Usage
//!
//! If a pointer passes through this camera's render target, it will
//! automatically shoot rays into the rapier scene and will be able to pick things.
//!
//! To ignore an entity, you can add [`Pickable::IGNORE`] to it, and it will be ignored during
//! raycasting.
//!
//! For fine-grained control, see the [`RapierBackendSettings::require_markers`] setting.
//!
//! ## Limitations
//!
//! Because raycasting is expensive, only the closest intersection will be reported. This means that
//! unlike some UI, you cannot hover multiple rapier objects with a single pointer by configuring
//! the [`Pickable`] component to not block lower elements but still emit events. As mentioned
//! above, all that is supported is completely ignoring an entity with [`Pickable::IGNORE`].
//!
//! This is probably not a meaningful limitation, as the feature is usually only used in UI where
//! you might want a pointer to be able to pick multiple elements that are on top of each other. If
//! are trying to build a UI out of rapier entities, beware, I suppose.

#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![deny(missing_docs)]

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_math::Vec3;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{prelude::*, view::RenderLayers};
use bevy_transform::prelude::*;
use bevy_window::PrimaryWindow;

use bevy_picking_core::backend::prelude::*;
use bevy_rapier3d::{na::{Const, OPoint}, prelude::*};

// Re-export for uses who want this
pub use bevy_rapier3d;

/// Commonly used imports.
pub mod prelude {
    pub use crate::{RapierBackend, RapierBackendSettings, RapierPickable};
}

/// Adds the `rapier` raycasting picking backend to your app.
#[derive(Clone)]
pub struct RapierBackend;
impl Plugin for RapierBackend {
    fn build(&self, app: &mut App) {
        app.init_resource::<RapierBackendSettings>()
            .add_systems(PreUpdate, update_hits.in_set(PickSet::Backend))
            .register_type::<RapierBackendSettings>()
            .register_type::<RapierPickable>();
    }
}

/// Runtime settings for the [`RapierBackend`].
#[derive(Resource, Default, Reflect)]
#[reflect(Resource, Default)]
pub struct RapierBackendSettings {
    /// When set to `true` raycasting will only happen between cameras and entities marked with
    /// [`RapierPickable`]. Off by default. This setting is provided to give you fine-grained
    /// control over which cameras and entities should be used by the rapier backend at runtime.
    pub require_markers: bool,
}

/// Optional. Marks cameras and target entities that should be used in the rapier picking backend.
/// Only needed if [`RapierBackendSettings::require_markers`] is set to true.
#[derive(Debug, Clone, Default, Component, Reflect)]
#[reflect(Component, Default)]
pub struct RapierPickable;

/// Raycasts into the scene using [`RapierBackendSettings`] and [`PointerLocation`]s, then outputs
/// [`PointerHits`].
pub fn update_hits(
    pointers: Query<(&PointerId, &PointerLocation)>,
    primary_window_entity: Query<Entity, With<PrimaryWindow>>,
    picking_cameras: Query<(
        Entity,
        &Camera,
        &GlobalTransform,
        Option<&RapierPickable>,
        Option<&RenderLayers>,
    )>,
    pickables: Query<&Pickable>,
    marked_targets: Query<&RapierPickable>,
    layers: Query<&RenderLayers>,
    backend_settings: Res<RapierBackendSettings>,
    q_colliders: Query<(&Collider, &GlobalTransform)>,
    rapier_context: Option<Res<RapierContext>>,
    mut output_events: EventWriter<PointerHits>,
) {
    let Some(rapier_context) = rapier_context else {
        return;
    };

    for (pointer_id, pointer_location) in &pointers {
        let pointer_location = match pointer_location.location() {
            Some(l) => l,
            None => continue,
        };
        for (cam_entity, camera, ray, cam_layers) in picking_cameras
            .iter()
            .filter(|(_, camera, ..)| {
                camera.is_active && pointer_location.is_in_viewport(camera, &primary_window_entity)
            })
            .filter(|(.., marker, _)| marker.is_some() || !backend_settings.require_markers)
            .filter_map(|(entity, camera, transform, _, layers)| {
                let mut viewport_pos = pointer_location.position;
                if let Some(viewport) = &camera.viewport {
                    viewport_pos -= camera.logical_viewport_rect().unwrap().min;
                }
                camera
                    .viewport_to_world(transform, viewport_pos)
                    .map(|ray| (entity, camera, ray, layers))
            })
        {
            // if we hit a backface, start a new trace just in front of the surface we hit
            // we can do this to skip backfaces when picking
            let mut ray_start = ray.origin;
            let mut collected_toi = 0.0;
            for iteration in 0..2 {
                if let Some((entity, hit_data)) = rapier_context
                    .cast_ray_and_get_normal(
                        ray_start,
                        ray.direction,
                        f32::MAX,
                        true,
                        QueryFilter::new().predicate(&|entity| {
                            let marker_requirement =
                                !backend_settings.require_markers || marked_targets.get(entity).is_ok();

                            // Cameras missing render layers intersect all layers
                            let cam_layers = cam_layers.copied().unwrap_or(RenderLayers::all());
                            // Other entities missing render layers are on the default layer 0
                            let entity_layers = layers.get(entity).copied().unwrap_or_default();
                            let render_layers_match = cam_layers.intersects(&entity_layers);

                            let pickable = pickables
                                .get(entity)
                                .map(|p| *p != Pickable::IGNORE)
                                .unwrap_or(true);
                            marker_requirement && render_layers_match && pickable
                        }),
                    )
                    .and_then(|(entity, hit)| {
                        if let Ok((collider, global_transform)) = q_colliders.get(entity) {
                            let point = OPoint::<f32, Const<3>>::new(hit.point.x, hit.point.y, hit.point.z);
                            if let Some(surface_normal) = collider.raw.feature_normal_at_point(hit.feature, &point) {
                                // transform the surface normal back into world space
                                let transformed_surface_normal = global_transform.compute_matrix().transform_vector3(surface_normal.into());
                                let is_backface = ray.direction.normalize_or_zero().dot(transformed_surface_normal) > 0.0;
                                if is_backface {
                                    ray_start = hit.point + ray.direction * f32::EPSILON;
                                    collected_toi += hit.toi;
                                    return None;
                                }
                            }
                        }

                        let hit_data =
                            HitData::new(cam_entity, collected_toi + hit.toi, Some(hit.point), Some(hit.normal));
                        Some((entity, hit_data))
                    })
                {
                    output_events.send(PointerHits::new(
                        *pointer_id,
                        vec![(entity, hit_data)],
                        camera.order as f32,
                    ));
                    break;
                }
            }
        }
    }
}
