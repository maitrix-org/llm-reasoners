

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(planet a)
(planet b)
(craves c d)
(craves d a)
(province b)
(province c)
)
(:goal
(and
(craves a b)
(craves b d)
(craves d c))
)
)


