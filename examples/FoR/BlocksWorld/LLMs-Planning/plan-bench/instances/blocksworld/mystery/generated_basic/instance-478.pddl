

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(craves a b)
(planet b)
(craves c d)
(craves d e)
(craves e a)
(province c)
)
(:goal
(and
(craves a d)
(craves b e)
(craves e a))
)
)


