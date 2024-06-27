(define (problem depot-3-1-3-4-4-3) (:domain depots)
(:objects
	depot0 depot1 depot2 - Depot
	distributor0 - Distributor
	truck0 truck1 truck2 - Truck
	pallet0 pallet1 pallet2 pallet3 - Pallet
	crate0 crate1 crate2 - Crate
	hoist0 hoist1 hoist2 hoist3 - Hoist)
(:init
	(at pallet0 depot0)
	(clear pallet0)
	(at pallet1 depot1)
	(clear crate0)
	(at pallet2 depot2)
	(clear crate2)
	(at pallet3 distributor0)
	(clear pallet3)
	(at truck0 distributor0)
	(at truck1 distributor0)
	(at truck2 depot0)
	(at hoist0 depot0)
	(available hoist0)
	(at hoist1 depot1)
	(available hoist1)
	(at hoist2 depot2)
	(available hoist2)
	(at hoist3 distributor0)
	(available hoist3)
	(at crate0 depot1)
	(on crate0 pallet1)
	(at crate1 depot2)
	(on crate1 pallet2)
	(at crate2 depot2)
	(on crate2 crate1)
)

(:goal (and
		(on crate0 pallet1)
		(on crate2 pallet2)
	)
))
