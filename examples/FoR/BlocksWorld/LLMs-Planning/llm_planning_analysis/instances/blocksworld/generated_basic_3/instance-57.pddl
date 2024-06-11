

(define (problem BW-rand-3)
(:domain blocksworld-4ops)
(:objects a b c )
(:init
(handempty)
(on a c)
(on b a)
(ontable c)
(clear b)
)
(:goal
(and
(on a b)
(on b c))
)
)


