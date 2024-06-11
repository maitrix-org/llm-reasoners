(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects g e d j)
(:init 
(harmony)
(planet g)
(planet e)
(planet d)
(planet j)
(province g)
(province e)
(province d)
(province j)
)
(:goal
(and
(craves g e)
(craves e d)
(craves d j)
)))