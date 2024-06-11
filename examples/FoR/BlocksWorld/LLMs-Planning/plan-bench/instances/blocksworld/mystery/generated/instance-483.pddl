(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects g e i)
(:init 
(harmony)
(planet g)
(planet e)
(planet i)
(province g)
(province e)
(province i)
)
(:goal
(and
(craves g e)
(craves e i)
)))