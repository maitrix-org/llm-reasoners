(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects l d g i)
(:init 
(harmony)
(planet l)
(planet d)
(planet g)
(planet i)
(province l)
(province d)
(province g)
(province i)
)
(:goal
(and
(craves l d)
(craves d g)
(craves g i)
)))