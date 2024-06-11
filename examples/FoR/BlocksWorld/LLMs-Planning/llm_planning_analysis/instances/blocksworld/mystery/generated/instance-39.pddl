(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects h e i)
(:init 
(harmony)
(planet h)
(planet e)
(planet i)
(province h)
(province e)
(province i)
)
(:goal
(and
(craves h e)
(craves e i)
)))