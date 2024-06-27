

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(craves a e)
(planet b)
(planet c)
(craves d c)
(craves e d)
(province a)
(province b)
)
(:goal
(and
(craves a b)
(craves c a)
(craves d e)
(craves e c))
)
)


