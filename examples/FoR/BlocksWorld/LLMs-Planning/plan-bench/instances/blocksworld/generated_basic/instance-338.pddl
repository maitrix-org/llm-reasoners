

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(on a d)
(on b a)
(ontable c)
(on d c)
(clear b)
)
(:goal
(and
(on a d)
(on d b))
)
)


